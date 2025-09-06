#%load_ext autoreload
import pandas as pd
import numpy as np
import os, re, json, time, ast
import dotenv
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union, Literal, Optional, Any
from pydantic import BaseModel
from dataclasses import dataclass, field, asdict  # does not force type validation unlike pydantic
from sqlalchemy import create_engine, Engine

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)

from rich.console import Console
from rich.markdown import Markdown
console = Console()


from helpers import (
    paper_supplies, 
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
    pretty_print_financial_report
)


# Set up, load env parameters and instantiate LLM
dotenv.load_dotenv()
openai_api_key = os.getenv('UDACITY_OPENAI_API_KEY')

model = OpenAIServerModel(
    model_id='gpt-4o-mini',
    api_base='https://openai.vocareum.com/v1',
    api_key=openai_api_key,
)



########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


@tool
def generate_order_id() -> str:
    """Generate a unique order ID."""
    state.order_counter += 1
    return f"ORD-{state.order_counter:04d}"

@tool
def extract_request_date(text: str) -> str:
    """
    Extracts the 'Date of request' from an order string.
    Returns a datetime.date if found, else None.

    Args:
        text: customer order with section 'Date of request: date to extract in the format "%Y-%m-%d"'

    Returns
        extracted date (str) or None
    """
    match = re.search(r'Date of request:\s*([0-9]{4}[-/][0-9]{2}[-/][0-9]{2})', text)
    if match:
        try:
            return match.group(1)
        except ValueError:
            return None
    return None

@tool
def count_order_quantities(text: str) -> tuple[int, list[int]]:
    """
    Tentatively extract the number of stock items from customer order and the requested quantities

    Return (count, quantities_list) where quantities_list are integers that
    represent ordered item quantities. The function:
      - accepts numbers with thousands separators (e.g., 10,000)
      - ignores numbers inside dates (ISO-like and 'Month Day, Year' forms)
      - ignores digits attached to letters (e.g., A4, A3, X-12)
      - ignores decimal numbers (e.g. 3.5)
      - ignores inches, percentages and weight/length related numbers (e.g. 20 lb, 50% or 24")

    Args:
        text (str): customer order text to analyze
    
    Returns:
        tuple[int, list[int]: the count of extracted quantities and their list
    """

    # Month names pattern for natural-language dates
    MONTHS = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|' \
         r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|' \
         r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'

    exclude_spans = []

    # --- Exclude dates (collect their spans) ---
    # ISO-like YYYY-MM-DD or YYYY/MM/DD
    for m in re.finditer(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', text):
        exclude_spans.append(m.span())

    # Year-Month (no day), e.g., 2025-04 or 2025/04
    for m in re.finditer(r'\b\d{4}[-/]\d{1,2}\b', text):
        exclude_spans.append(m.span())

    # Month Day, Year  (e.g., April 15, 2025)  and  'Apr 15 2025'
    for m in re.finditer(rf'\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*|\s+)\d{{4}}\b',
                         text, flags=re.IGNORECASE):
        exclude_spans.append(m.span())

    # Day Month Year  (e.g., 15 April 2025)
    for m in re.finditer(rf'\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}\s+\d{{4}}\b',
                         text, flags=re.IGNORECASE):
        exclude_spans.append(m.span())

    # Month Year  (e.g., April 2025)
    for m in re.finditer(rf'\b{MONTHS}\s+\d{{4}}\b', text, flags=re.IGNORECASE):
        exclude_spans.append(m.span())

    # Numeric M/D/Y or D/M/Y (e.g., 04/15/2025, 15/04/2025)
    for m in re.finditer(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', text):
        exclude_spans.append(m.span())

    # --- Exclude digits that are part of codes like A4, A3, X-12, etc. ---
    for m in re.finditer(r'\b[A-Za-z]+-\d+\b|\b[A-Za-z]+\d+[A-Za-z]*\b', text):
        s, e = m.span()
        sub = text[s:e]
        for d in re.finditer(r'\d+', sub):  # add only the digits inside that token
            exclude_spans.append((s + d.start(), s + d.end()))

    # --- Exclude decimals ---
    for m in re.finditer(r'\b\d+\.\d+\b', text):  # e.g., 3.5
        exclude_spans.append(m.span())

    # --- Exclude inches (24", 30”, 40in, 24') ---
    for m in re.finditer(r'\b\d+(?:,\d{3})*(?=\s*(?:"|”|\'|in\b))', text, flags=re.IGNORECASE):
        exclude_spans.append(m.span())

    # --- Exclude percentages (50%) ---
    for m in re.finditer(r'\b\d+(?:,\d{3})*(?=%)', text):
        exclude_spans.append(m.span())

    # --- Exclude weight/length units (30 lb, 20 kg, 5 g, 10 oz, etc.) ---
    for m in re.finditer(r'\b\d+(?:,\d{3})*\s*(lb|lbs|kg|g|oz|mg|m|cm|mm)\b',
                         text, flags=re.IGNORECASE):
        exclude_spans.append(m.span())


    def is_in_excluded(span):
        a, b = span
        for x, y in exclude_spans:
            if a < y and b > x:  # overlaps any excluded region
                return True
        return False

    # --- Pick candidate numbers (not adjacent to letters), allow 1,234,567 ---
    quantities = []
    for m in re.finditer(r'(?<![A-Za-z])(\d{1,3}(?:,\d{3})+|\d+)(?![A-Za-z])', text):
        if not is_in_excluded(m.span()):
            quantities.append(int(m.group(1).replace(',', '')))

    return len(quantities), quantities




class OrderProcessorAgent(ToolCallingAgent):
    """Agent responsible for processing customer order requests."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[],
            model=model,
            name="order_processor",
            description="Agent responsible for processing customer orders. Parses orders, identifies requested products and quantities."
        )



# Tools for inventory agent
@tool
def evaluate_purchase_requirement(item_name:str, quantity_required:int, input_date_str:str, db_engine: Engine=db_engine)->str:
    """
    compares the quantity on stock at a specific date to the quantity from the BOM
    - generate a sales transaction in the system
    - generate a stock order transaction to replenish the inventory when required
    - collect the availaility date for ordered materials
    logs all the transaction ids and availability date for later use
    
    Args:
        item_name (stock_item): The name of the item to look up. One of ['100 lb cover stock','80 lb text paper','A4 paper','Banner paper','Butcher paper','Cardstock','Colored paper','Crepe paper','Glossy paper','Invitation cards','Kraft paper','Large poster paper (24x36 inches)','Paper plates','Patterned paper','Photo paper','Presentation folders','Rolls of banner paper (36-inch width)','Table covers']
        quantity_required (int): The number of units from the BOM.
        input_date_str (str): Date of the order for the look up in ISO format (YYYY-MM-DD).
        db_engine (Engine): sqlite database client (default = db_engine)            
    """

    global state

    # first we check that the order date was properly captured, otherwise we use state.order_date if it exists
    if state.order_date:
        if input_date_str!=state.order_date:
            input_date_str = state.order_date

    # check current inventory level
    current_stock_level = get_stock_level(item_name, input_date_str).at[0,'current_stock']

    min_stock_level = state.min_stock_level[item_name]
    sales_id = None
    order_id=None

    # the quantity is in stock - can be supplied immediately
    if (current_stock_level - quantity_required) >= min_stock_level:
        availability_date = input_date_str

    else:
        quantity_to_order = min_stock_level + quantity_required - current_stock_level

        if quantity_required <= current_stock_level:
            # we have enough to supply immediatly
            availability_date = input_date_str
        else:
            # we need to replenish first
            availability_date = get_supplier_delivery_date(input_date_str=input_date_str, quantity=quantity_to_order)

        # stock replenishement transac
        order_id = create_transaction(
            item_name=item_name, 
            date=input_date_str, # CONVENTION: purchase order issued at the date of the customer order
            quantity=quantity_to_order,
            transaction_type='stock_orders',
            price= quantity_to_order * state.price_list[item_name]
        )

        # update state with purchase order id
        state.pending_transactions.append(order_id)

    # update with best availability date in all scanerios    
    state.availability_date.append(availability_date)

    # Sale transac
    sales_id = create_transaction(
            item_name=item_name, 
            date=input_date_str,  # CONVENTION: invoices and Sales accounted for when order received #state.requested_date
            quantity=quantity_required,
            transaction_type='sales',
            price= quantity_required * state.price_list[item_name]
        )

    # update state with sales id
    state.pending_transactions.append(sales_id)

    # confirm transactions
    message = f"{quantity_required} {item_name} confirmed with availability date: {availability_date}"
    if sales_id:
         message += f" - sales transaction id: {sales_id} (quantity: {quantity_required}, value: {quantity_required * state.price_list[item_name]}, due date: {input_date_str})"
    if order_id:
        message += f" - stock order transaction id: {order_id} (quantity: {quantity_to_order}, value: {quantity_to_order * state.price_list[item_name]}, due date: {input_date_str})"  

    #//////////////// CONTROL \\\\\\\\\\\\\\\\\
    report = generate_financial_report(input_date_str)
    pretty_print_financial_report(report, version='short')
    #//////////////////////////////////////////
    
    return message



class InventoryManagerAgent(ToolCallingAgent):
    """Agent responsible for managing ingredients inventory."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                #get_all_inventory,
                #get_stock_level,
                evaluate_purchase_requirement,
            ],
            model=model,
            name="inventory_manager",
            description=f"""
You are the Inventory Manager for the Beaver paper company.
You receive a bill of materials prepared by the Agent analyzing a customer order.

<task>
Your job is to:
1. Check inventory and place stock orders (purchase orders) necessary to supply a customer order (use 'evaluate_purchase_requirement' tool)
2. Answer any questions on the inventory
</task>

<steps>
1. Check what is required to deliver each stock item from the BOM (use 'evaluate_purchase_requirement' tool).
2. In case one of the requested stock item does not exist, repond the order cannot be completed.
3. confirm products availability, availability dates, the transaction ids.
</steps>
"""
        )


# Tools for quoting agent
@tool
def generate_quotation(bill_of_materials: list[dict]) -> str:
    """
    Calculate the total price of a list of stock items and quantities from a bill of materials.

    Args:
        bill_of_materials (list[dict]): Example format:
            [
                {"item": "Glossy paper", "quantity": 200},
                {"item": "Cardstock", "quantity": 100}
            ]

    Returns:
        str: Information on the total price for the bill_of_materials
    """
    if not bill_of_materials:
        return "Error: empty bill of materials - nothing to price"

    price = 0
    for entry in bill_of_materials:
        item = entry.get("item")
        quantity = entry.get("quantity", 0)

        if item not in state.available_items:
            return f"Cannot proceed: {item} not in available stock items"

        price += quantity * state.price_list[item]

    return f"The total price amounts to: {price}"


class QuotingAgent(ToolCallingAgent):
    """Agent responsible for providing quotation of customer order requests."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[generate_quotation],
            model=model,
            name="quotation_agent",
            description="Agent responsible for providing one quotation price for a bill of materials."
        )





# Set up your agents and create an orchestration agent that will manage them.
class BeaverOrchestrator:
    """Coordinates the multi-agent Beaver company system."""
    
    def __init__(self, model):
        self.order_processor = OrderProcessorAgent(model)
        self.inventory_manager = InventoryManagerAgent(model)
        self.quotation_agent = QuotingAgent(model)

    def analyze_request(self, customer_request):
        """
        Process a customer order from initial request through order confirmation.
        
        Args:
            customer_request: Natural language order request from customer with requested delivery date
            
        Returns:
            Response to customer with order details and status
        """

        #################################################################
        ################## STEP 1: Parse the order ######################
        #################################################################

        #    example: If the customer asks for 'A4 glossy paper', select the available 'glossy paper' and not 'A4 paper'.

        # Naive count of requested quantities in the order to help avoiding double count
        count, quantities = count_order_quantities(customer_request)

        order_processing_base_prompt =f"""The customer says: "{customer_request}"
<intructions>
Your taks is to extract from the order:
- Each stock item (type, size, grade and finish) and associated quantity
- the requested date of delivery (%Y-%m-%d isoformat)
- success: True if the order can be fulfilled, False otherwise
- comment: brief justification on changes or unavailabilities

Hint: preliminary analysis says the customer asks for {count} stock items and quantities of {quantities}.
</intructions>

<Process>
- Analyze the order carefully and extract each stock item (type, size, grade and finish). One requested stock item can be extracted only once.
- find the corresponding stock item label to use in the valid inventory references {state.available_items}
- If one stock item does not exist, check for a valid equivalent replacement.
- If one stock item cannot be found, add to the BOM, success is False
- summarize in the comment section the replacement made (success true) or the unavailable product(s) (success false)
</Process>

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
"Glossy paper: 500, matte A3 paper: 1000, Large poster paper (24x36 inches): 300, Cardstock: 200, delivery date:2025-04-15, success:false, comment: 'A3 paper are not inventory items'"
-----------
Order:"
I would like to place an order for the following paper supplies for the reception: 
- 200 sheets of A4 white printer paper
- 100 sheets of A3 glossy paper
- 50 packets of 100% recycled kraft paper envelopes
I need these supplies delivered by April 10, 2025. Thank you. (Date of request: 2025-04-07)"
Output:
"A4 paper: 200, Glossy paper: 100, 100% recycled kraft envelopes: 50, delivery date:2025-04-10, success:false, comment: 'envelopes are not inventory items'"
</examples>

"""     
        max_retries = 3
        retries = 0
        reprocess = True

        order_processing_prompt = order_processing_base_prompt

        while reprocess and retries < max_retries:
            
            # call the agent
            response1 = self.order_processor.run(order_processing_prompt + "\n\nReturn one consolidated analysis for the customer order.")

            # parse agent response using parsing func
            parsed_response = parse_agent_response(
                response1,
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
            print_rich(f"CONTROL PARSING check1= {not check1}, check2= {not check2}, check3= {not check3}", color='bright_green')
            #//////////////////////////////////////////

            # if agent already rejected the order then jump to early stopping here
            if not parsed_response['success']:
                reprocess = False
                break
            
            # if the agent validates the order but the BOM has detected issues then retry loop with feedback
            reprocess = check1 or check2 or check3
            if reprocess:
                print_rich(f"Reprocess needed, attempt {retries+1}/{max_retries}", color="orange3")
                retries += 1

                order_processing_prompt = (order_processing_base_prompt + 
                    f"\nThere was an error in your previous answer: {bill_of_materials}")
                if check1:
                    order_processing_prompt += f"\nThese stock items do not exist in the allowed list: {error_list}."
                if check2:
                    order_processing_prompt += "\nYour bill of materials has too many items (double-counting)."
                if check3:
                    order_processing_prompt += f"\nYour bill of materials has {len(bill_of_materials)} items when {count} are expected."
                order_processing_prompt += " Please retry."

            else:
                print_rich("All stock items are valid, proceeding...", color="orange3")


        # Early stopping: max retries reached and extraction failure not resolved
        if reprocess:
            print_rich("Max retries reached — one or more item is still invalid or double-counted quantities.", color="bright_red")
            if error_list:
                parsed_response['comment'] += " We do not have these products: " + " ,".join(error_list)
            return ("We were not able to process your order. "
                    f"Comment: {parsed_response['comment']}")


        #//////////////// CONTROL \\\\\\\\\\\\\\\\\
        print_rich(json.dumps(parsed_response, indent=2), color='bright_green')
        #//////////////////////////////////////////


        # Early stopping success False condition
        if not parsed_response['success']:
            message = f"Unfortunately we cannot validate your order."
            if error_list:
                message+=f" We do not have these products: {', '.join(error_list)}"
            message+=f" {parsed_response['comment']}"
            return message

        
        # Generate one customer order ID using generate_order_id for this customer order
        id = generate_order_id()
        parsed_response['id'] = id

        # capture date of order
        if extract_request_date(customer_request):
            order_date = extract_request_date(customer_request)
            state.order_date = order_date
        else:
            print_rich("Error extracting date of order", color='bright_red')

        # capture requested delivery date
        state.requested_date = parsed_response['delivery_date']




        #################################################################
        ######### STEP 2: Manage inventory and transactions #############
        #################################################################

        inventory_manager_prompt = f"""The order processing agent prepared this bill of materials for you:

{json.dumps(parsed_response['bill of materials'])}

Order date: {order_date}
requested delivery date: {parsed_response['delivery_date']}

Process the bill of materials
"""
        
        response2 = self.inventory_manager.run(inventory_manager_prompt)
        
        #//////////////// CONTROL \\\\\\\\\\\\\\\\\
        #print_rich(f"CONTROL TRANSACTIONS STATE: {state.pending_transactions}", color='bright_red')
        if state.order_date:
            print_rich('============================================================================', color='orange3')
            report = generate_financial_report(state.order_date)
            pretty_print_financial_report(report, version='full')
            print_rich('============================================================================', color='orange3')
        #//////////////////////////////////////////



        #################################################################
        ################# STEP 3: Provide Quotation #####################
        #################################################################

        quotation_agent_prompt = f"""The order processing agent prepared this bill of materials for you:
{json.dumps(parsed_response['bill of materials'])}

Return the total price for the bill of materials.
"""

        response3 = self.quotation_agent.run(quotation_agent_prompt)
        # Regex for integers and floats (with optional decimal part)
        pattern = r"\d+\.\d+|\d+"
        matches = re.findall(pattern, response3)
        floats = [float(m) for m in matches]
        if floats:
            order_price = floats[0]
        else:
            order_price = response3

        #//////////////// CONTROL \\\\\\\\\\\\\\\\\
        #print_rich(f"CONTROL PRICE CALC = {response3}", color='bright_green')
        #//////////////////////////////////////////


        ##################################################################
        ### STEP 4: Check availability date vs requested delivery date ###
        ##################################################################

        requested_date = state.requested_date # str format
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
            return (f"Your order {id} with requested delivery date {requested_date} is confirmed."
            f" Total price for this order is {order_price} credits."
            f" You can pick up your supplies starting {best_available_date}."
            f" Additional comment: {parsed_response['comment']}"
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

        return (f"The price for your order amounts to {order_price} credits."
        f" Unfortunatly we cannot confirm the delivery date of {requested_date}."
        f" Your order {id} can be ready by {availability_date} at the earliest due to supply delays on some products."
        f" Additional comment: {parsed_response['comment']}"
        " LET US KNOW IF THIS AVAILABILITY DATE IS ACCEPTABLE FOR YOU SO THAT WE CAN PROCESS YOUR ORDER."
        )





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


    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    # Initialize State with key parameters
    state.initial_date = initial_date

    orchestrator = BeaverOrchestrator(model)
    print_rich("orchestrator initialized")


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


        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        # IMPORTANT: Reset state at the start of each cycle
        state.customer_request = request_with_date
        state.pending_transactions = []
        state.availability_date = []
        state.customer_order_id = None
        state.requested_date = None
        state.order_date = None

        # trigger agentic workflow with orchestrator.analyze_request method
        response = orchestrator.analyze_request(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print_rich(f"Response: {response}", color="bright_magenta")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

        """if idx>3:
            break # ...........................................................
"""
    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    print("test results file saved")
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":    
    results = run_test_scenarios()