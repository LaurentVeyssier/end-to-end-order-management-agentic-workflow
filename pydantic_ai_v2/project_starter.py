#%load_ext autoreload
import pandas as pd
import os, re, json, time
import dotenv
from sqlalchemy.sql import text
from datetime import datetime
from typing import Dict, List, Union, Literal, Optional, Any
from pydantic import BaseModel
from dataclasses import dataclass, field, asdict  # does not force type validation unlike pydantic
from sqlalchemy import create_engine, Engine

import nest_asyncio
nest_asyncio.apply()

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from rich.console import Console
from rich.markdown import Markdown
console = Console()

from helpers import (
    generate_sample_inventory, 
    init_database, 
    db_engine,
    state
    )

from utils import (
    print_rich,
    create_transaction,
    get_all_inventory,
    get_stock_level,
    get_supplier_delivery_date,
    get_cash_balance,
    generate_financial_report,
    search_quote_history,
    parse_agent_response,
    delete_transaction,
    pretty_print_financial_report,
    generate_order_id,
    count_order_quantities,
    evaluate_purchase_requirement,
    generate_quotation,
    response_to_customer,
)


# Set up, load env parameters and instantiate LLM
dotenv.load_dotenv()
openai_api_key = os.getenv('UDACITY_OPENAI_API_KEY')


model = OpenAIChatModel(
    'gpt-4o-mini',
    provider=OpenAIProvider(
        base_url='https://openai.vocareum.com/v1', api_key=openai_api_key,
    ),
)




class OrderProcessorAgent(Agent):
    """Agent responsible for processing customer order requests."""
    
    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            tools=[],
            model=model,
            name="order_processor",
            system_prompt=("You are an experienced customer service agent taking customer orders for a paper supply company BeaverPaper. "
            "you are responsible for processing customer orders. Parses orders, identifies requested products and quantities.")
        )


class InventoryManagerAgent(Agent):
    """Agent responsible for managing ingredients inventory."""
    
    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            tools=[
                #get_all_inventory,
                #get_stock_level,
                evaluate_purchase_requirement,
            ],
            model=model,
            name="inventory_manager",
            system_prompt=f"""
You are the Inventory Manager for the BeaverPaper company.
You process a bill of materials prepared by the Agent analyzing a customer order or you answer general inquiry.

<task>
Your job is to:
1. Bill of Materials: Check inventory and place stock orders (purchase orders) necessary to supply a customer order (use 'evaluate_purchase_requirement' tool)
2. Inquiry: Answer any questions on the inventory
</task>

<steps>
for a bill of materials:
1. Check what is required to deliver each stock item from the BOM (use 'evaluate_purchase_requirement' tool).
2. In case one of the requested stock item does not exist, repond the order cannot be completed.
3. confirm products availability, availability dates, the transaction ids.

for an inquiry:
1. Answer the inquiry
</steps>
"""
        )


class QuotingAgent(Agent):
    """Agent responsible for providing quotation of customer order requests."""
    
    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            tools=[generate_quotation],
            model=model,
            retries=2,
            name="quotation_agent",
            system_prompt="You are a quotation analyst at BeaverPaper company, responsible for providing one quotation price for a bill of materials."
        )

    
class AdvisorAgent(Agent):
    """Agent answering questions about Beaver financials."""
    
    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            tools=[get_cash_balance, generate_financial_report, search_quote_history],
            model=model,
            retries=1,
            end_strategy="early",
            name="advisor_agent",
            system_prompt="You are a financial analyst at BeaverPaper company, responsible for answering finance-related questions and provide advice."
        )

    def process_query(self, query:str):

        advisor_prompt = f"""Using the available tools, answer this query: {query}
        No discount are allowed. Never use 'rebate' in your search.
        Today is {state.order_date}
        """

        response = self.run_sync(advisor_prompt)

        return response.output


class RouterAgent(Agent):
    """Agent responsible to route the incoming request to the appropriate agent."""
    
    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            tools=[],
            model=model,
            retries=2,
            name="router_agent",
            system_prompt="You determine the appropriate agent to process an incoming request."
        )

    def route_request(self, request:str) -> Literal['process_order','inventory_agent','advisor_agent']:

        router_prompt = f"""You receive this request: {request}.

Choose the approapriate agent to process the request.
- process_order: the request is a customer order
- inventory_agent: the request is about inventory or stock item price
- advisor_agent: cash, financial metrics, top selling products, quotes history, rebate or discounts
- inventory_agent: the request is a general inquiry
"""
        response = self.run_sync(router_prompt).output

        for valid_response in ['process_order','inventory_agent','advisor_agent']:
            if valid_response in response.lower():
                route_to_agent = valid_response
                return route_to_agent

        # fallback position
        return 'inventory_agent'



# Set up your agents and create an orchestration agent that will manage them.
class BeaverOrchestrator:
    """Coordinates the multi-agent Beaver company system."""
    
    def __init__(self, model):
        self.order_processor = OrderProcessorAgent(model)
        self.inventory_manager = InventoryManagerAgent(model)
        self.quotation_agent = QuotingAgent(model)
        self.router_agent = RouterAgent(model)
        self.advisor_agent = AdvisorAgent(model)

    def routing_request(self, customer_request):

        route_to_agent = self.router_agent.route_request(customer_request)
        
        match route_to_agent:
            case 'process_order':
                route_to_agent = self.analyze_request
            case 'inventory_agent':
                route_to_agent = self.process_inventory
            case 'advisor_agent':
                route_to_agent = self.advisor_agent.process_query

        return route_to_agent

    #################################################################
    ################## STEP 1: Parse the order ######################
    #################################################################

    def analyze_request(self, customer_request):
        """
        Process a customer order from initial request through order confirmation.
        
        Args:
            customer_request: Natural language order request from customer with requested delivery date
            
        Returns:
            Response to customer with order details and status
        """

        # Naive count of requested quantities in the order to help avoiding double count
        count, quantities = count_order_quantities(customer_request)

        order_processing_base_prompt =f"""
The customer says: "{customer_request}"
<intructions>
Your taks is to extract from the order:
- Each stock item (type, size, grade and finish) and associated quantity
- the requested date of delivery (%Y-%m-%d isoformat)
- success: True if the order can be fulfilled, False otherwise
- comment: brief justification on changes or unavailabilities

Hint: preliminary analysis says the customer asks for {count} stock items and quantities of {quantities}.
</intructions>

<Process>
- Analyze the order carefully and extract each stock item. One requested stock item can be extracted only once.
- For consistence, use the name of the corresponding stock item in the valid inventory references {state.available_items}
- If one stock item cannot be found, use a reasonably equivalent replacement of similar type in the available inventory references.
- If one stock item cannot be found or replaced, report label as-is in the BOM, with success False, otherwise success is True (all items can be supplied).
- summarize in the comment section the replacement(s) made and the unavailable item(s).
</Process>

<rules>
- References with a finish but no size are available in all sizes (e.g. Glossy paper reference is available in A4, A3 or A5).
- References are available in different colors unless mentionned (e.g. Poster paper is available in various colors).
</rules>

<Response>
Respond using enumeration stock item name:quantity, stock item name:quantity,...requested delivery date, comment and success (True or False).

stock item name:quantity,
stock item name:quantity,
[...]
delivery date: requested date,
sucess: true/false,
comment: short comment informing about alternative or non existing stock item, leave empty otherwise
</Response>

<examples>
Order:
"I would like to place a large order for various types of paper supplies for our upcoming exhibition. We need the 
following items:
- 500 sheets of glossy A4 paper
- 1000 sheets of matte A3 paper
- 300 poster boards (24" x 36")
- 200 sheets of heavyweight cardstock
We need these supplies delivered by April 15, 2025. Thank you. (Date of request: 2025-04-07)"
Output:
Glossy paper: 500, Matte paper: 1000, Large poster paper (24x36 inches): 300, 250 gsm cardstock: 200, delivery date:2025-04-15, success:true, comment: Glossy paper is available in A4, Matte paper in A3, our 250 gsm cardstock corresponds to the heavyweight requirement.
-----------
Order:"
I would like to place an order for the following paper supplies for the reception: 
- 200 sheets of A4 white printer paper
- 100 sheets of A3 glossy paper
- 50 packets of 100% recycled kraft paper envelopes
I need these supplies delivered by April 10, 2025. Thank you. (Date of request: 2025-04-07)"
Output:
A4 paper: 200, Glossy paper: 100, 100% recycled kraft paper envelopes: 50, delivery date:2025-04-10, success:false, comment: We can replace A4 white printer paper with simple A4 paper but we do not have recycled kraft paper envelopes, only plain envelopes.
</examples>

"""     
        state.order_counter+=1
        max_retries = 3
        retries = 0
        reprocess = True

        order_processing_prompt = order_processing_base_prompt

        while reprocess and retries < max_retries:
            
            # call the agent
            response1 = self.order_processor.run_sync(order_processing_prompt + "\n\nReturn one consolidated analysis for the customer order.")

            # parse agent response using parsing func
            parsed_response = parse_agent_response(
                response1.output,
                customer_request,
                ALLOWED_ITEMS=state.available_items
            )

            # extract bill of materials
            bill_of_materials = parsed_response['bill of materials']

            # Check if all items are valid or not double counted
            #check1 = not all(item['item'] in state.available_items for item in bill_of_materials)
            error_list =[]
            for item in bill_of_materials:
                item_value = item.get('item') or "undetermined"
                if item_value not in state.available_items:
                    error_list.append(item_value)

            check1 = len(error_list) > 0                # we have extracted invalid stock items (ex: item not found in stock references)
            check2 = len(bill_of_materials) > count     # we have too many extracted items (ex: double-counting 'A4 glossy' -> 'A4' and 'Glossy')
            check3 = len(bill_of_materials) < count     # we are missing some requested items (ex: non valid items are excluded by LLM)

            #//////////////// CONTROL \\\\\\\\\\\\\\\\\
            print_rich(f"CONTROL PARSING check1= {not check1}, check2= {not check2}, check3= {not check3}, success= {parsed_response['success']}", color='bright_green')
            #//////////////////////////////////////////

            reprocess = check1 or check2 or check3
            order_processing_prompt = (order_processing_base_prompt + 
                    f"\nThere was an error in your previous answer:"
                    f"\n{bill_of_materials}"
                    f"\ncomment: {parsed_response['comment']}"
                    f"\nsuccess: {parsed_response['success']}"
                    )

            if not check1 and not parsed_response['success']:
                # we have an incoherence: success failure but all BOM items are valid
                reprocess = True
                print_rich("inconsistency detected between failed success flag but valid BOM", color='orange3')
                order_processing_prompt += f"\nInconsistency detected: You report success false but all BOM items are valid stock references BeaverPaper can supply."

            # the agent rejected the order (success=false) and check1 found not exiting items then jump to early stopping here
            elif not parsed_response['success']:
                reprocess = False
                break
            
            # if the agent validates the order but the BOM has detected issues then retry loop with feedback
            if reprocess:
                print_rich(f"Reprocess needed, attempt {retries+1}/{max_retries}", color="orange3")
                retries += 1
                if check1:
                    order_processing_prompt += f"\nThese stock items do not exist as stock reference: {error_list}."
                if check2:
                    order_processing_prompt += "\nYour bill of materials has too many items (double-counting)."
                if check3:
                    order_processing_prompt += f"\nYour bill of materials has {len(bill_of_materials)} items when {count} are expected."
                
                print_rich(f"Feedback provided as context to agent: {order_processing_prompt.split(order_processing_base_prompt)[1]}", color="orange3")
                
                order_processing_prompt += " Please retry."

            else:
                print_rich("All stock items are valid, proceeding...", color="orange3")


        # Early stopping: max retries reached and extraction failure not resolved
        if reprocess:
            print_rich("Max retries reached — one or more item is still invalid or double-counted quantities.", color="bright_red")
            return response_to_customer(
                decision = "We were not able to process your order.",
                status = "ORDER NOT CONFIRMED",
                reason = f"We do not have these products: {', '.join(error_list)}" if error_list else "",
                comment = f"{parsed_response['comment']}",
            )


        #//////////////// CONTROL \\\\\\\\\\\\\\\\\
        print_rich(json.dumps(parsed_response, indent=2), color='bright_green')
        #//////////////////////////////////////////


        # Early stopping success False condition
        if not parsed_response['success']:
            return response_to_customer(
                decision = "Unfortunately we cannot validate your order.",
                status = "ORDER NOT CONFIRMED",
                reason = f"We do not have these products: {', '.join(error_list)}" if error_list else "",
                comment = f"{parsed_response['comment']}",
            )

        
        # Generate one customer order ID using generate_order_id for this customer order
        id = generate_order_id()
        parsed_response['id'] = id

        # capture order level key information, e.g. BOM, requested delivery date
        state.delivery_date = parsed_response['delivery_date']
        state.bill_of_materials = parsed_response['bill of materials']
        state.comment = parsed_response['comment']
        state.customer_order_id = id


    #################################################################
    ######### STEP 2: Manage inventory and transactions #############
    #################################################################

    def process_inventory(self, text_input:str=None) -> str:
        
        if not text_input:
            inventory_manager_prompt = f"""The order processing agent prepared this bill of materials for you:

{json.dumps(state.bill_of_materials)}

Order date: {state.order_date}
Requested delivery date: {state.delivery_date}

Process the bill of materials
"""

            response2 = self.inventory_manager.run_sync(inventory_manager_prompt)
            
            #//////////////// CONTROL \\\\\\\\\\\\\\\\\
            #print_rich(f"CONTROL TRANSACTIONS STATE: {state.pending_transactions}", color='bright_red')
            '''if state.order_date:
                print_rich('============================================================================', color='orange3')
                report = generate_financial_report(state.order_date)
                pretty_print_financial_report(report, version='full')
                print_rich('============================================================================', color='orange3')'''
            #//////////////////////////////////////////

        else:
            # answer general inquiry
            response2 = self.inventory_manager.run_sync(text_input).output

        return response2


    #################################################################
    ################# STEP 3: Provide Quotation #####################
    #################################################################

    def process_sales(self, text_input:str=None) -> str:


        if not text_input:

            quotation_agent_prompt = f"""The order processing agent prepared this bill of materials for you:
{json.dumps(state.bill_of_materials)}

Return the total price for the bill of materials.
"""
            response3 = self.quotation_agent.run_sync(quotation_agent_prompt)
            # Regex for integers and floats (with optional decimal part)
            pattern = r"\d+\.\d+|\d+"
            matches = re.findall(pattern, response3.output)
            floats = [float(m) for m in matches]
            if floats:
                order_price = floats[0]
            else:
                order_price = response3.output

            #//////////////// CONTROL \\\\\\\\\\\\\\\\\
            #print_rich(f"CONTROL PRICE CALC = {response3.output}", color='bright_green')
            #//////////////////////////////////////////

            state.total_price = order_price

        else:
            # answer general inquiry
            response3 = self.quotation_agent.run_sync(text_input).output
            return response3

        ##################################################################
        ### STEP 4: Check availability date vs requested delivery date ###
        ##################################################################

    def confirm_order(self) -> str:

        requested_date = state.delivery_date # str format
        # set availability_date to the the requested date from customer (worst case)
        availability_date = requested_date
        best_available_date = requested_date

        if state.availability_date:
            # check if each availability date is before requested date
            for due_date in state.availability_date:
                if pd.to_datetime(availability_date) > pd.to_datetime(due_date):
                    availability_date = due_date
                if pd.to_datetime(due_date) < pd.to_datetime(best_available_date):   
                    best_available_date = due_date 

        #//////////////// CONTROL \\\\\\\\\\\\\\\\\
        #print_rich(f"CONTROL DATES => Requested {requested_date}, Available: {availability_date}, Best available: {best_available_date}, State: {state.availability_date}", color='bright_green')
        #//////////////////////////////////////////

        if pd.to_datetime(availability_date) <= pd.to_datetime(requested_date):

            return response_to_customer(
                decision = f"Your order {state.customer_order_id} with requested delivery date {requested_date} is confirmed.",
                status = "ORDER CONFIRMED",
                comment = f"{state.comment}",
                availability_date=best_available_date,
                delivery_date=requested_date,
                price=state.total_price
            )

        # the order cannot be confirmed due to availability date > requested delivery date
        # reverse the orders in the db
        if state.pending_transactions:
            for id in state.pending_transactions:
                deleted = delete_transaction(id)
                if deleted:
                    print_rich(f"Transaction {id} deleted successfully", color="bright_red")
                    # update state
                    state.order_counter=-1
                else:
                    print_rich(f"No transaction found with that id = {id}", color="bright_red")

            # update state
            state.pending_transactions = []

        return response_to_customer(
                decision = f"Unfortunately your order {state.customer_order_id} with requested delivery date {requested_date} has not been validated.",
                status = "ORDER NOT CONFIRMED",
                reason= f"We cannot confirm the delivery date of {requested_date}.\nLET US KNOW IF THIS AVAILABILITY DATE IS ACCEPTABLE FOR YOU SO THAT WE CAN PROCESS YOUR ORDER.",
                comment = f"{state.comment}",
                availability_date=availability_date,
                delivery_date=requested_date,
                price=state.total_price
            )


    def final_response(self, response:str)->str:
        
        # Update state
        report = generate_financial_report(state.order_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print_rich(Markdown(f"Response: {response}"), color="bright_magenta")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        time.sleep(1)

        return {
                "request_date": state.order_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response.replace('\n',' '),
            }


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print_rich(f"FATAL: Error loading test data: {e}",color="bright_red")
        return

    quote_requests_sample = pd.read_csv("quote_requests_sample.csv")

    to_add = pd.DataFrame([
        {"job":"office manager", "need_size":"small", "event":"party", 
        "request":"Can I have a discount for a very large order of A4 paper?",
        "request_date":datetime(2025,4,5)},
        {"job":"office manager", "need_size":"small", "event":"party", 
        "request":"what would be the price for 1000 A4 paper sheets?",
        "request_date":datetime(2025,4,5)},
        { "job":"office manager", "need_size":"small", "event":"party", 
        "request":"what products can I order?",
        "request_date":datetime(2025,4,3)},
        {"job":"office manager", "need_size":"small", "event":"party", 
        "request":"what is the price of A3 paper?",
        "request_date":datetime(2025,4,4)},
        {"job":"office manager", "need_size":"small", "event":"party", 
        "request":"do you have stock for 1000 flyers?",
        "request_date":datetime(2025,4,5)},
        {"job":"office manager", "need_size":"small", "event":"party", 
        "request":"what is our cash position?",
        "request_date":datetime(2025,4,5)},
        {"job":"office manager", "need_size":"small", "event":"party", 
        "request":"Find all past quotes with a discount (not rebate)? Summarize in a table",
        "request_date":datetime(2025,4,5)},
        ]
    )

    quote_requests_sample = pd.concat([quote_requests_sample,to_add])

    # Sort by date
    quote_requests_sample["request_date"] = pd.to_datetime(
        quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
    )
    quote_requests_sample = quote_requests_sample.sort_values("request_date")

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    
    # Initialize State with key parameters
    state.initial_date = initial_date

    # Initialize orchestrator
    orchestrator = BeaverOrchestrator(model)
    print_rich("orchestrator initialized")

    # Run test samples
    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date:str = row["request_date"].strftime("%Y-%m-%d")  # request_date of type str

        print_rich(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        #//////////////// CONTROL \\\\\\\\\\\\\\\\\
        print_rich(request_with_date, color='bright_green')
        #/////////////////////////////////////////

        # IMPORTANT: Reset state at the start of each cycle
        state.customer_request = request_with_date
        state.order_date = request_date
        state.bill_of_materials = []
        state.comment=None
        state.pending_transactions = []
        state.total_price = None
        state.availability_date = []
        state.customer_order_id = None
        state.delivery_date = None

        result = {}
        result["request_id"] = idx + 1

        # trigger agentic workflow
        while True:

            # Route request to appropriate agent
            route_to = orchestrator.routing_request(request_with_date)
            
            # Analyze a customer order
            if route_to == orchestrator.analyze_request:
                response = orchestrator.analyze_request(request_with_date)
                # if response not None, this means early stopping (order rejected)
                if response:
                    result = orchestrator.final_response(response)
                    break
                
                # else continue to process inventory
                response = orchestrator.process_inventory()

            # Routing to handle inventory-related inquiries
            if route_to == orchestrator.process_inventory:
                prompt_inquiry = f"""Only use below information to answer this inquiry. Do NOT use tools.

                Available references:
                {state.available_items}

                Available stocks:
                {get_all_inventory(state.order_date)}

                Price list:
                {state.price_list}

                Inquiry: {request_with_date}

                response:
                """
                response = orchestrator.process_inventory(prompt_inquiry)
                result = orchestrator.final_response(response)
                break

            # Routing to handle finance inquiries
            if route_to == orchestrator.advisor_agent.process_query:
                response = orchestrator.advisor_agent.process_query(request_with_date)
            
            # customer order processing quotation step
            if route_to == orchestrator.analyze_request:
                response = orchestrator.process_sales()
            
            # if response not None, this means last step of the response to an inquiry
            if response:
                result = orchestrator.final_response(response)
                break

            # customer order processing confirmation step
            response = orchestrator.confirm_order()

            # customer order processing final response step
            result = orchestrator.final_response(response)
            break

        # append result to results list
        results.append(result)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    print("\ntest results file saved")
    pd.DataFrame(results).to_csv("test_results.csv", index=False, sep=';')  #..................................................
    return results


if __name__ == "__main__":    
    results = run_test_scenarios()