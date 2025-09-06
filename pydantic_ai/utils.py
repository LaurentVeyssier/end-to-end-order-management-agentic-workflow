import pandas as pd
import numpy as np
import os,re,json
import time
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union, Literal
from sqlalchemy import create_engine, Engine

from rich.console import Console
from rich.markdown import Markdown
console = Console()

from helpers import db_engine,state

# Given below are some utility functions you can use to implement your multi-agent system

def print_rich(text:str, color:str='bright_yellow')->None:
    """
    improved print command adding color and markdown support using rich library
    inputs:
        text: text to print
        color: rich-color to use
    outputs:
        None
    
    """
    console.print(text, style=f"bold {color}")


def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """

    print(f"FUNC (create_transaction): Executing {transaction_type} order on {item_name} for qty {quantity} x {price} = {quantity*price:.2f} for '{date}'")

    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # get last transaction_id
        id = state.transaction_counter   #........ADDED to auto-increment id parameter

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "id": id,                         #........ADDED to auto-increment id parameter
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # update state
        state.transaction_counter+=1  #........ADDED to auto-increment id parameter

        with db_engine.begin() as conn:            #........ADDED to control auto-increment id parameter during parallel executions
            # Insert the record into the database
            transaction.to_sql("transactions", conn, if_exists="append", index=False)

            # Fetch and return the ID of the inserted row
            result = pd.read_sql("SELECT last_insert_rowid() as id", conn)
            return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise



def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))



def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.
    
    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )



def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")



def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def pretty_print_financial_report(report: Dict, version: Literal['short','full']) -> None:
    """printing of financial report for improved visualization"""
    report_date = report['as_of_date']
    cash_balance = report['cash_balance']
    inventory_value = report['inventory_value']
    total_assets = report['total_assets']
    inventory_summary = pd.DataFrame(report['inventory_summary'])
    top_selling_products = pd.DataFrame(report['top_selling_products'])

    with console.capture() as capture:
        console.print(f"[orange3]FINANCIAL REPORT as of {report_date}[/]")
        console.print(f"[orange3]- cash balance    : {cash_balance:.2f}[/]")
        console.print(f"[orange3]- inventory value : {inventory_value:.2f}[/]")
        console.print(f"[orange3]- total assets    : {total_assets:.2f}[/]")

        if version == 'full':
            console.print(f"[orange3]-------- Inventory summary --------[/]")
            console.print(inventory_summary)
            console.print(f"[orange3]------- Top selling products ------[/]")
            console.print(top_selling_products)

    # print captured block all at once
    print(capture.get())
    


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row) for row in result]



def parse_agent_response(response: str, order:str, ALLOWED_ITEMS:List):
    """parse orderprocessor agent response to extract all usefull information"""
    
    result = {
        "success":None, 
        "bill of materials":[{"item": None, "quantity": None}], 
        "delivery_date": None, 
        "comment": None, 
        "customer order":order
        }

    # --- Extract success ---
    success_match = re.search(r"success:\s*(.*),", response, re.IGNORECASE)
    if success_match:
        success = success_match.group(1).strip()
        result["success"] = success.lower()=="true" if success else False

    # --- Extract delivery date ---
    date_match = re.search(r"delivery date:\s*([\d-]+)", response, re.IGNORECASE)
    if date_match:
        try:
            extracted_date = date_match.group(1)
            result["delivery_date"] = datetime.strptime(extracted_date, "%Y-%m-%d").date().isoformat()
        except ValueError:
            pass  # leave as None if invalid

    # --- Extract comment ---
    comment_match = re.search(r"comment:\s*(.*)", response, re.IGNORECASE)
    if comment_match:
        comment = comment_match.group(1).strip()
        result["comment"] = comment if comment else None

    # --- Extract items ---
    # Split by commas but ignore parts with 'delivery date' or 'comment'
    parts = [p.strip() for p in response.split(",") if not re.search(r"(delivery date|comment)", p, re.IGNORECASE)]
    
    bill_of_materials = []
    for part in parts:
        # match "Item:Quantity"
        match = re.match(r"(.+?):\s*(\d+)", part)
        if match:
            item, qty = match.groups()
            item = item.strip()
            #if item in ALLOWED_ITEMS:
            bill_of_materials.append({"item":item, "quantity":int(qty)})
            #break  # only one allowed
    
    if bill_of_materials:
        result["bill of materials"] = bill_of_materials

    return result



def delete_transaction(transaction_id: int) -> bool:
    """
    Delete a transaction from the 'transactions' table by its id.

    Args:
        transaction_id (int): The id of the transaction to delete.

    Returns:
        bool: True if a row was deleted, False if no matching row was found.
    """
    print(f"FUNC (delete_transaction): Reversing transaction {transaction_id}")
    try:
        with db_engine.begin() as conn:
            result = conn.execute(
                text("DELETE FROM transactions WHERE id = :id"),
                {"id": transaction_id}
            )
            return result.rowcount > 0  # True if a row was deleted
    except Exception as e:
        print(f"Error deleting transaction {transaction_id}: {e}")
        raise


if __name__ == "__main__":
    pass



def generate_order_id() -> str:
    """Generate a unique order ID."""
    state.order_counter += 1
    return f"ORD-{state.order_counter:04d}"


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



# Tools for inventory agent
def evaluate_purchase_requirement(item_name:str, quantity_required:int, input_date_str:str)->str:
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
    #report = generate_financial_report(input_date_str)
    #pretty_print_financial_report(report, version='short')
    #//////////////////////////////////////////
    
    return message


# Tools for quoting agent
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
    
    print(f"FUNC (generate_quotation): Calculating total price of a bill of materials")
    
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



def response_to_customer(
    decision:str,
    status:str,
    reason:str = '',
    comment:str = '',
    price:float = None,
    delivery_date:str = '',
    availability_date:str = '',
    ) ->str:
    """helper function to construct response to customer on order status"""
    message = decision
    message += f"\nReason: {reason}" if reason else ""
    message += f"\nStatus: {status}"
    message += f"\nAvailability date:{availability_date}" if availability_date else ""
    message += f"\nComment: {comment}" if comment else ""
    message += f"\nTotal price: {price}" if price else ""
    message += f"\nPayment due date: {delivery_date}" if price else ""
    return message