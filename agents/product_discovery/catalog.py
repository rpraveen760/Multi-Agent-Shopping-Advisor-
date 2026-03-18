"""Mock product catalog backend for PDF-aligned similar-product lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass

from common.a2a_models import Product, SimilarProductsRequest, SimilarProductsResponse


@dataclass(frozen=True)
class CatalogItem:
    name: str
    category: str
    price: str
    rating: str
    url: str
    source: str
    key_features: tuple[str, ...]
    tags: tuple[str, ...]


CATALOG_ITEMS: tuple[CatalogItem, ...] = (
    CatalogItem(
        name="Razer Viper V3 Pro",
        category="gaming mouse",
        price="$159.99",
        rating="4.7/5",
        url="https://catalog.example/products/razer-viper-v3-pro",
        source="Mock Catalog",
        key_features=("54g lightweight shell", "Focus Pro Gen 2 sensor", "Esports-first shape"),
        tags=("gaming", "mouse", "wireless", "esports", "lightweight", "razer"),
    ),
    CatalogItem(
        name="Razer DeathAdder V3 Hyperspeed",
        category="gaming mouse",
        price="$99.99",
        rating="4.6/5",
        url="https://catalog.example/products/razer-deathadder-v3-hyperspeed",
        source="Mock Catalog",
        key_features=("Ergonomic right-handed design", "Hyperspeed wireless", "Lightweight frame"),
        tags=("gaming", "mouse", "wireless", "ergonomic", "razer"),
    ),
    CatalogItem(
        name="Logitech G Pro X Superlight 2",
        category="gaming mouse",
        price="$149.99",
        rating="4.6/5",
        url="https://catalog.example/products/logitech-g-pro-x-superlight-2",
        source="Mock Catalog",
        key_features=("Superlight competitive shape", "Hero 2 sensor", "Long battery life"),
        tags=("gaming", "mouse", "wireless", "esports", "logitech", "lightweight"),
    ),
    CatalogItem(
        name="HyperX Pulsefire Haste 2 Wireless",
        category="gaming mouse",
        price="$89.99",
        rating="4.4/5",
        url="https://catalog.example/products/hyperx-pulsefire-haste-2-wireless",
        source="Mock Catalog",
        key_features=("Lightweight shell", "Dual wireless connectivity", "Responsive sensor"),
        tags=("gaming", "mouse", "wireless", "hyperx", "lightweight"),
    ),
    CatalogItem(
        name="Wacom Cintiq 16",
        category="drawing tablet",
        price="$649.95",
        rating="4.7/5",
        url="https://catalog.example/products/wacom-cintiq-16",
        source="Mock Catalog",
        key_features=("15.6-inch full HD pen display", "Natural pen feel", "Creative desktop workflow"),
        tags=("drawing", "tablet", "pen display", "wacom", "screen"),
    ),
    CatalogItem(
        name="XP-Pen Artist 13.3 Pro",
        category="drawing tablet",
        price="$299.99",
        rating="4.5/5",
        url="https://catalog.example/products/xp-pen-artist-13-3-pro",
        source="Mock Catalog",
        key_features=("13.3-inch laminated display", "Portable creative setup", "Affordable pen display"),
        tags=("drawing", "tablet", "pen display", "xp-pen", "screen"),
    ),
    CatalogItem(
        name="Huion Kamvas 16",
        category="drawing tablet",
        price="$399.00",
        rating="4.4/5",
        url="https://catalog.example/products/huion-kamvas-16",
        source="Mock Catalog",
        key_features=("15.6-inch display", "Battery-free pen", "Good value for artists"),
        tags=("drawing", "tablet", "pen display", "huion", "screen"),
    ),
    CatalogItem(
        name="NVIDIA RTX 5070",
        category="gpu",
        price="$549.00",
        rating="4.4/5",
        url="https://catalog.example/products/nvidia-rtx-5070",
        source="Mock Catalog",
        key_features=("1440p gaming target", "DLSS 4", "Ray tracing support"),
        tags=("gpu", "graphics card", "nvidia", "rtx", "gaming"),
    ),
    CatalogItem(
        name="NVIDIA RTX 5070 Ti",
        category="gpu",
        price="$749.00",
        rating="4.5/5",
        url="https://catalog.example/products/nvidia-rtx-5070-ti",
        source="Mock Catalog",
        key_features=("Stronger 1440p and entry 4K performance", "DLSS 4", "Ray tracing support"),
        tags=("gpu", "graphics card", "nvidia", "rtx", "gaming"),
    ),
    CatalogItem(
        name="AMD Radeon RX 9070 XT",
        category="gpu",
        price="$699.00",
        rating="4.4/5",
        url="https://catalog.example/products/amd-radeon-rx-9070-xt",
        source="Mock Catalog",
        key_features=("Competitive raster performance", "High VRAM", "Strong value positioning"),
        tags=("gpu", "graphics card", "amd", "radeon", "gaming"),
    ),
    CatalogItem(
        name="BestOffice Gaming Chair",
        category="gaming chair",
        price="$129.99",
        rating="4.1/5",
        url="https://catalog.example/products/bestoffice-gaming-chair",
        source="Mock Catalog",
        key_features=("Budget ergonomic build", "Lumbar pillow", "Reclining backrest"),
        tags=("gaming", "chair", "budget", "ergonomic"),
    ),
    CatalogItem(
        name="Homall Gaming Chair",
        category="gaming chair",
        price="$149.99",
        rating="4.2/5",
        url="https://catalog.example/products/homall-gaming-chair",
        source="Mock Catalog",
        key_features=("PU leather seat", "Retractable footrest", "Headrest support"),
        tags=("gaming", "chair", "budget", "ergonomic"),
    ),
    CatalogItem(
        name="Corsair TC100 Relaxed",
        category="gaming chair",
        price="$249.99",
        rating="4.5/5",
        url="https://catalog.example/products/corsair-tc100-relaxed",
        source="Mock Catalog",
        key_features=("Wide seat base", "Comfort-oriented cushioning", "Trusted gaming brand"),
        tags=("gaming", "chair", "ergonomic", "corsair"),
    ),
    CatalogItem(
        name="Sony WF-1000XM5",
        category="wireless earbuds",
        price="$299.99",
        rating="4.6/5",
        url="https://catalog.example/products/sony-wf-1000xm5",
        source="Mock Catalog",
        key_features=("Strong ANC", "Premium sound", "Compact flagship earbuds"),
        tags=("earbuds", "wireless", "anc", "sony", "audio"),
    ),
    CatalogItem(
        name="Bose QuietComfort Ultra Earbuds",
        category="wireless earbuds",
        price="$299.99",
        rating="4.5/5",
        url="https://catalog.example/products/bose-quietcomfort-ultra-earbuds",
        source="Mock Catalog",
        key_features=("Class-leading comfort", "Strong noise cancellation", "Immersive audio"),
        tags=("earbuds", "wireless", "anc", "bose", "audio"),
    ),
)


def _normalize_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}


def _infer_category(request: SimilarProductsRequest) -> str:
    if request.category:
        return request.category.strip().lower()

    haystack = " ".join([request.product_name, *request.features]).lower()
    category_hints = {
        "gaming mouse": ("mouse", "mice"),
        "drawing tablet": ("tablet", "pen display", "stylus", "artist"),
        "gpu": ("gpu", "graphics", "rtx", "radeon", "graphics card"),
        "gaming chair": ("chair", "lumbar", "ergonomic"),
        "wireless earbuds": ("earbuds", "earbud", "anc", "wireless buds"),
        "mechanical keyboard": ("keyboard", "switch", "keycap"),
    }
    for category, hints in category_hints.items():
        if any(hint in haystack for hint in hints):
            return category
    return "general"


def _extract_price_value(price: str | None) -> float | None:
    if not price:
        return None
    digits = re.sub(r"[^0-9.]", "", price)
    if not digits:
        return None
    try:
        return float(digits)
    except ValueError:
        return None


def _score_item(item: CatalogItem, request: SimilarProductsRequest, category: str) -> float:
    request_tokens = _normalize_tokens(request.product_name + " " + " ".join(request.features))
    item_tokens = _normalize_tokens(item.name + " " + " ".join(item.tags) + " " + " ".join(item.key_features))
    overlap = len(request_tokens & item_tokens)

    score = 0.0
    if item.category.lower() == category:
        score += 4.0

    score += overlap * 0.45

    request_price = _extract_price_value(request.price)
    item_price = _extract_price_value(item.price)
    if request_price and item_price:
        delta = abs(request_price - item_price) / max(request_price, 1)
        score += max(0.0, 1.2 - delta)

    if item.name.lower() == request.product_name.lower():
        score -= 3.0

    if any(tag in request_tokens for tag in ("wireless", "ergonomic", "lightweight", "budget", "competitive")):
        matching_tags = len(set(item.tags) & request_tokens)
        score += matching_tags * 0.2

    return score


def find_similar_products(
    request: SimilarProductsRequest,
    *,
    max_results: int = 3,
) -> SimilarProductsResponse:
    """Return stable structured similar products from the mock catalog."""
    category = _infer_category(request)
    ranked = sorted(
        (
            (_score_item(item, request, category), item)
            for item in CATALOG_ITEMS
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )

    products: list[Product] = []
    for score, item in ranked:
        if score <= 0:
            continue
        products.append(
            Product(
                name=item.name,
                price=item.price,
                rating=item.rating,
                url=item.url,
                key_features=list(item.key_features[:3]),
                source=item.source,
                evidence_urls=[item.url],
                confidence=min(0.98, round(0.5 + score / 10.0, 2)),
            )
        )
        if len(products) >= max_results:
            break

    summary = (
        f"Found {len(products)} similar products for {request.product_name} in the mock catalog."
        if products
        else f"No mock catalog matches were found for {request.product_name}."
    )
    return SimilarProductsResponse(request=request, products=products, summary=summary)
