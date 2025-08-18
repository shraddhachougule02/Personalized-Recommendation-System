import sys
import os
import time
import ast
from typing import List, Optional, Tuple, Any, Dict

from fastapi import FastAPI, Form, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.gzip import GZipMiddleware
import anyio

# ---------- Bootstrapping ----------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from services.auth import validate_user
from services import search as semantic_search_mod
from services.recommend import (
    recommend_items_hybrid,
    get_product_details_for_recommendations,
    user_item_matrix,
    user_similarity,
    item_similarity,
    cleaned_data,
)

MIN_PRICE = 10700
MAX_PRICE = 3930000

# ---------- Tiny TTL cache ----------
_CACHE: Dict[str, Tuple[float, Any]] = {}

def cache_get(key: str) -> Optional[Any]:
    item = _CACHE.get(key)
    if not item:
        return None
    exp, val = item
    if exp < time.time():
        _CACHE.pop(key, None)
        return None
    return val

def cache_set(key: str, val: Any, ttl: int = 120):
    _CACHE[key] = (time.time() + ttl, val)

# ---------- Helpers ----------
def _safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def _parse_listish(cell):
    """
    Returns a Python list of strings from a cell that may contain:
    - a real list (['A','B'])
    - a stringified list "['A','B']"  or "('A','B')"
    - a comma-separated string "A,B"
    - a single scalar "A"
    - NaN/None
    """
    if cell is None:
        return []
    # handle pandas NaN
    if isinstance(cell, float) and str(cell) == "nan":
        return []
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    if isinstance(cell, str):
        s = cell.strip()
        # try list/tuple literal
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, (list, tuple)):
                    return [str(x).strip() for x in val if str(x).strip()]
            except Exception:
                pass
        # comma-separated
        if "," in s:
            return [part.strip() for part in s.split(",") if part.strip()]
        return [s] if s else []
    return [str(cell).strip()]

# ---------- App ----------
app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=500)

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# ---------- helper to render the user home (so we can call from POST and from GET) ----------
async def render_existing_user_home(request: Request, user_id: str) -> HTMLResponse:
    if user_id.isdigit():
        user_id = f"user_{int(user_id):03d}"
    if not validate_user(user_id):
        return HTMLResponse(
            "<h3>User ID not found. Try again or register as a new user.</h3>"
            "<a href='/existing_user'>Try Again</a><br>"
            "<a href='/'>Back to Home</a>"
        )
    recommended_ids = await anyio.to_thread.run_sync(
        recommend_items_hybrid,
        user_id, user_item_matrix, user_similarity, item_similarity,
        5, 0.5
    )
    rec_df = await anyio.to_thread.run_sync(
        get_product_details_for_recommendations, recommended_ids, cleaned_data
    )
    rec_df = rec_df.loc[:, ["product_name", "brand", "price"]]
    return templates.TemplateResponse(
        "existing_user_home.html",
        {
            "request": request,
            "user_id": user_id,
            "recommended_products": rec_df.to_dict(orient="records"),
        }
    )

# ---------- Pages ----------
@app.get("/", response_class=HTMLResponse)
async def welcome(request: Request):
    return templates.TemplateResponse("welcome.html", {"request": request})

@app.get("/existing_user", response_class=HTMLResponse)
async def existing_user(request: Request):
    return templates.TemplateResponse("existing_user.html", {"request": request})

@app.get("/new_user", response_class=HTMLResponse)
async def new_user(request: Request):
    return templates.TemplateResponse("new_user.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, user_id: str = Form(...)):
    return await render_existing_user_home(request, user_id)

# GET route to return to the logged-in home (used by results back-link)
@app.get("/existing_user_home", response_class=HTMLResponse)
async def existing_user_home(request: Request, user_id: str = Query(...)):
    return await render_existing_user_home(request, user_id)

# Dedicated search page (optional)
@app.get("/search_page", response_class=HTMLResponse)
async def search_page(request: Request, query: Optional[str] = Query(None)):
    return templates.TemplateResponse("search_page.html", {"request": request, "query": query or ""})

# Results page (SEMANTIC SEARCH) – accepts user_id for correct back link
@app.get("/search_results", response_class=HTMLResponse)
async def search_results(
    request: Request,
    query: str = Query(...),
    top_k: int = 15,
    user_id: Optional[str] = Query(None),
):
    q = (query or "").strip()
    if not q:
        return templates.TemplateResponse("search_results.html", {
            "request": request, "query": q, "results": [], "user_id": user_id
        })
    cache_key = f"search_results::{q}::{top_k}"
    cached = cache_get(cache_key)
    if cached is None:
        def do_search():
            return semantic_search_mod.search_products(q, {}, top_k)
        df = await anyio.to_thread.run_sync(do_search)
        cols = [c for c in ["product_name", "brand", "price"] if c in df.columns]
        results = df.loc[:, cols].to_dict(orient="records")
        cache_set(cache_key, results, ttl=180)
    else:
        results = cached

    return templates.TemplateResponse("search_results.html", {
        "request": request,
        "query": q,
        "results": results,
        "user_id": user_id,
    })

# ---------- OPTIONS for dropdowns (robust) ----------
@app.get("/options")
async def get_filter_options():
    # product types (scalar)
    product_types = (
        cleaned_data["product_type"].dropna().astype(str).unique().tolist()
        if "product_type" in cleaned_data.columns else []
    )
    product_types = sorted(set(product_types))

    # brands (scalar)
    brands = (
        cleaned_data["brand"].dropna().astype(str).unique().tolist()
        if "brand" in cleaned_data.columns else []
    )
    brands = sorted(set(brands))

    # skin types (list-like)
    skin_types_set = set()
    if "skintype_list" in cleaned_data.columns:
        for v in cleaned_data["skintype_list"].dropna():
            skin_types_set.update(_parse_listish(v))
    skin_types = sorted(x for x in skin_types_set if x)

    # skin concerns: accept multiple possible column names
    concern_cols = ["skin_concern", "skin_concern_list", "notable_effects", "notable_effects_list"]
    skin_concerns_set = set()
    for col in concern_cols:
        if col in cleaned_data.columns:
            for v in cleaned_data[col].dropna():
                skin_concerns_set.update(_parse_listish(v))
            break  # use the first column that exists
    skin_concerns = sorted(x for x in skin_concerns_set if x)

    return {
        "product_types": product_types,
        "brands": brands,
        "skin_types": skin_types,
        "skin_concerns": skin_concerns,
        "min_price": MIN_PRICE,
        "max_price": MAX_PRICE,
    }

# ---------- APIs (used by ajax search if needed) ----------
@app.get("/search")
async def search_api(
    query: str = Query(..., description="Free-text semantic query"),
    top_k: int = Query(10, ge=1, le=100),
):
    q = query.strip()
    if not q:
        return JSONResponse(content=[])

    cache_key = f"semsearch_api::{q}::{top_k}"
    cached = cache_get(cache_key)
    if cached is not None:
        return JSONResponse(content=cached)

    def _run():
        return semantic_search_mod.search_products(q, {}, top_k)

    df = await anyio.to_thread.run_sync(_run)
    cols = [c for c in ["product_name", "brand", "price"] if c in df.columns]
    out = df.loc[:, cols].to_dict(orient="records")

    cache_set(cache_key, out, ttl=120)
    return JSONResponse(content=out)

# ---------- PAGE: Filter Results (robust parsing + overlap for list-like cols) ----------
@app.get("/filter_results", response_class=HTMLResponse)
async def filter_results_page(
    request: Request,
    product_type: Optional[str] = Query(None, description="Face Wash/Toner/Serum/Moisturizer/Sunscreen"),
    skin_concern: Optional[List[str]] = Query(None, description="0+ concerns"),
    skintype_list: Optional[List[str]] = Query(None, description="0+ skin types"),
    brand: Optional[List[str]] = Query(None, description="0+ brands"),
    price_min: Optional[str] = Query(None),
    price_max: Optional[str] = Query(None),
    top_k: int = Query(20, ge=1, le=100),
    source: Optional[str] = Query(None),   # "existing_user" or "new_user"
    user_id: Optional[str] = Query(None),  # for existing_user back link
):
    # Back link
    if source == "existing_user" and user_id:
        back_to = f"/existing_user_home?user_id={user_id}"
    elif source == "existing_user":
        back_to = "/existing_user"
    else:
        back_to = "/new_user"

    # Price parsing + guards
    pm = _safe_float(price_min, None)
    px = _safe_float(price_max, None)
    price_min_val = max(pm, MIN_PRICE) if pm is not None else MIN_PRICE
    price_max_val = min(px, MAX_PRICE) if px is not None else MAX_PRICE
    if price_min_val > price_max_val:
        return templates.TemplateResponse(
            "filter_results.html",
            {"request": request, "results": [], "error_msg": "Invalid price range", "back_to": back_to},
        )

    # Normalize selections
    chosen_concerns = set((skin_concern or []))
    chosen_types    = set((skintype_list or []))
    chosen_brands   = set((brand or []))

    # Decide which column to use for concerns (first that exists)
    concern_cols = ["skin_concern", "skin_concern_list", "notable_effects", "notable_effects_list"]
    concern_col = next((c for c in concern_cols if c in cleaned_data.columns), None)

    def _filter_df():
        df = cleaned_data.copy()

        if product_type:
            df = df[df["product_type"] == product_type]

        if chosen_concerns and concern_col:
            df = df[df[concern_col].apply(lambda v: bool(set(_parse_listish(v)) & chosen_concerns))]

        if chosen_types and "skintype_list" in df.columns:
            df = df[df["skintype_list"].apply(lambda v: bool(set(_parse_listish(v)) & chosen_types))]

        if chosen_brands and "brand" in df.columns:
            df = df[df["brand"].isin(chosen_brands)]

        if "price" in df.columns:
            df = df[(df["price"] >= price_min_val) & (df["price"] <= price_max_val)]

        return df.head(top_k)

    products = await anyio.to_thread.run_sync(_filter_df)
    results = (
        products.loc[:, ["product_name", "brand", "price"]].to_dict(orient="records")
        if not products.empty else []
    )

    return templates.TemplateResponse(
        "filter_results.html",
        {
            "request": request,
            "results": results,
            "error_msg": None if results else "No products found for these filters.",
            "back_to": back_to,
        },
    )

# ---------- Also keep /price_range if you use it elsewhere ----------
@app.get("/price_range")
async def price_range():
    return {"min_price": MIN_PRICE, "max_price": MAX_PRICE}
 