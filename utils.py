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

from smolagents import tool

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

@tool
def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
    db_engine:Engine=db_engine
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
        db_engine (Engine): sqlite database client (default = db_engine)

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """

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


@tool
def get_all_inventory(as_of_date: str,db_engine:Engine=db_engine) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.
        db_engine (Engine): sqlite database client (default = db_engine)

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


@tool
def get_stock_level(item_name: str, as_of_date: Union[str, datetime], db_engine: Engine=db_engine) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.
    
    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.
        db_engine (Engine): sqlite database client (default = db_engine)

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


@tool
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


@tool
def get_cash_balance(as_of_date: Union[str, datetime], db_engine: Engine=db_engine) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.
        db_engine (Engine): sqlite database client (default = db_engine)

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

@tool
def generate_financial_report(as_of_date: Union[str, datetime], db_engine: Engine=db_engine) -> Dict:
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
        db_engine (Engine): sqlite database client (default = db_engine)

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
    

@tool
def search_quote_history(search_terms: List[str], limit: int = 5, db_engine: Engine=db_engine) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.
        db_engine (Engine): sqlite database client (default = db_engine)

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
            if item in ALLOWED_ITEMS:
                bill_of_materials.append({"item":item, "quantity":int(qty)})
                #break  # only one allowed
    
    if bill_of_materials:
        result["bill of materials"] = bill_of_materials

    return result


@tool
def delete_transaction(transaction_id: int, db_engine: Engine = db_engine) -> bool:
    """
    Delete a transaction from the 'transactions' table by its id.

    Args:
        transaction_id (int): The id of the transaction to delete.
        db_engine (Engine): sqlite database client (default = db_engine)

    Returns:
        bool: True if a row was deleted, False if no matching row was found.
    """
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