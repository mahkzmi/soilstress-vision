# api_unified.py
# ABAK 2.0 - Unified Soil Analysis API + Profit Prediction
# ادغام کامل: پیش‌بینی خصوصیات خاک + سود و زیان اقتصادی
# تاریخ: می ۲۰۲۶

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import joblib
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import math

# ============================================================
# راه‌اندازی اپلیکیشن
# ============================================================

app = FastAPI(
    title="ABAK 2.0 - Unified Soil Analysis API", 
    description="پیش‌بینی EC, pH, OM, بافت خاک, کربن و سودآوری محصولات",
    version="2.0.0"
)

# CORS برای اتصال به فرانت‌اند
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# بخش ۱: بارگذاری مدل‌های هوش مصنوعی (روز ۱)
# ============================================================

try:
    model_ec = joblib.load('model_ec.pkl')
    model_ph = joblib.load('model_ph.pkl')
    model_om = joblib.load('model_om.pkl')
    model_texture = joblib.load('model_texture.pkl')
    texture_mapping = joblib.load('texture_mapping.pkl')
    print("✅ مدل‌های هوش مصنوعی با موفقیت بارگذاری شدند")
except Exception as e:
    print(f"⚠️ خطا در بارگذاری مدل‌ها: {e}")
    print("لطفاً ابتدا train_all_models.py را اجرا کنید")
    model_ec = None
    model_ph = None
    model_om = None
    model_texture = None
    texture_mapping = {'sandy': 0, 'sandy_loam': 1, 'loam': 2, 'clay_loam': 3, 'clay': 4}

reverse_texture_mapping = {v: k for k, v in texture_mapping.items()}

# ============================================================
# بخش ۲: داده‌های محصولات (از ABAK قبلی)
# ============================================================

# عملکرد پایه در شرایط ایده‌آل (تن در هکتار)
CROP_BASE_YIELD = {
    'wheat': 5.0,
    'barley': 4.5,
    'corn': 8.0,
    'canola': 3.5,
    'sugar_beet': 55.0,
    'pistachio': 3.0,
    'saffron': 0.35,
    'cotton': 4.0,
    'alfalfa': 8.0,
    'sunflower': 2.5
}

# بخش: داده‌های محصولات - قیمت‌ها به‌روزرسانی شدند

# قیمت محصولات (تومان بر کیلوگرم) - بر اساس نرخ‌های واقعی سال ۱۴۰۴-۱۴۰۵
CROP_PRICE = {
    'wheat': 48500,        # گندم - قیمت تضمینی جدید
    'barley': 35000,       # جو
    'corn': 38000,         # ذرت
    'canola': 55000,       # کلزا
    'sugar_beet': 12000,   # چغندر قند
    'pistachio': 450000,   # پسته - صادراتی
    'saffron': 150000000,  # زعفران - کیلویی ۱۵۰ میلیون تومان
    'cotton': 75000,       # پنبه
    'alfalfa': 15000,      # یونجه
    'sunflower': 50000     # آفتابگردان
}

# هزینه تولید (تومان در هکتار) - به‌روز شده با توجه به افزایش قیمت نهاده‌ها
CROP_COST = {
    'wheat': 65000000,      # گندم (افزایش هزینه)
    'barley': 55000000,     # جو
    'corn': 75000000,       # ذرت
    'canola': 70000000,     # کلزا
    'sugar_beet': 90000000, # چغندر قند
    'pistachio': 120000000, # پسته
    'saffron': 180000000,   # زعفران (هزینه بالای داشت)
    'cotton': 80000000,     # پنبه
    'alfalfa': 50000000,    # یونجه
    'sunflower': 60000000   # آفتابگردان
}
# ضرایب کاهش عملکرد بر اساس شوری (FAO Salt Tolerance)
SALINITY_COEFF = {
    'wheat': {'threshold': 6.0, 'slope': 0.15},
    'barley': {'threshold': 8.0, 'slope': 0.10},
    'corn': {'threshold': 1.7, 'slope': 0.25},
    'canola': {'threshold': 5.0, 'slope': 0.20},
    'sugar_beet': {'threshold': 7.0, 'slope': 0.15},
    'pistachio': {'threshold': 10.0, 'slope': 0.08},
    'saffron': {'threshold': 7.0, 'slope': 0.12},
    'cotton': {'threshold': 7.0, 'slope': 0.12},
    'alfalfa': {'threshold': 5.0, 'slope': 0.10},
    'sunflower': {'threshold': 4.0, 'slope': 0.18}
}

# آستانه pH برای هر محصول
PH_OPTIMAL = {
    'wheat': (6.0, 8.0),
    'barley': (5.5, 8.5),
    'corn': (6.0, 7.5),
    'canola': (6.0, 7.5),
    'sugar_beet': (6.5, 7.8),
    'pistachio': (7.0, 8.5),
    'saffron': (7.0, 8.0),
    'cotton': (6.0, 8.0),
    'alfalfa': (6.5, 7.5),
    'sunflower': (6.0, 7.5)
}

# نیاز آبی (متر مکعب در هکتار)
WATER_REQUIREMENT = {
    'wheat': 4500,
    'barley': 4000,
    'corn': 6000,
    'canola': 3500,
    'sugar_beet': 7000,
    'pistachio': 8000,
    'saffron': 3500,
    'cotton': 5500,
    'alfalfa': 7000,
    'sunflower': 4500
}

# ============================================================
# بخش ۳: توابع پیش‌بینی خصوصیات خاک (از مدل‌های ML)
# ============================================================

def predict_soil_properties(ndvi, precipitation, temperature, elevation, clay):
    """پیش‌بینی EC, pH, OM, بافت خاک از روی داده‌های محیطی"""
    
    features = np.array([[
        ndvi, precipitation, temperature, elevation, clay
    ]])
    
    # پیش‌بینی‌ها با مدل‌های ML (با fallback در صورت نبود مدل)
    if model_ec is not None:
        ec = float(model_ec.predict(features)[0])
    else:
        ec = 2.0 + (clay / 50) + (temperature / 20)
    
    if model_ph is not None:
        ph = float(model_ph.predict(features)[0])
    else:
        ph = 7.0 - (temperature - 15) / 20 + (clay / 100)
    
    if model_om is not None:
        om = float(model_om.predict(features)[0])
    else:
        om = 1.5 + (ndvi * 2) + (clay / 50)
    
    # محدود کردن به بازه منطقی
    ec = max(0.2, min(16.0, ec))
    ph = max(3.0, min(10.0, ph))
    om = max(0.1, min(5.0, om))
    
    # پیش‌بینی بافت خاک
    if model_texture is not None:
        texture_code = int(model_texture.predict(features)[0])
        texture = reverse_texture_mapping.get(texture_code, 'loam')
    else:
        if clay > 40:
            texture = 'clay'
        elif clay > 30:
            texture = 'clay_loam'
        elif clay > 15:
            texture = 'loam'
        elif clay > 5:
            texture = 'sandy_loam'
        else:
            texture = 'sandy'
    
    # محاسبه کربن (فرمول استاندارد)
    carbon = (0.5 * ndvi * 10) + (0.01 * precipitation / 100) - (0.08 * (temperature - 10)) - (0.0005 * elevation) + (0.02 * clay / 10)
    carbon = max(0.5, min(8.0, carbon))
    
    return {
        "ec": round(ec, 2),
        "ph": round(ph, 2),
        "organic_matter": round(om, 2),
        "texture": texture,
        "carbon": round(carbon, 2)
    }

# ============================================================
# بخش ۴: توابع پیش‌بینی عملکرد و سود (از ABAK قبلی)
# ============================================================

def predict_yield(crop_type, ec, ph=None, om=None):
    """پیش‌بینی عملکرد بر اساس شوری و نوع محصول"""
    base_yield = CROP_BASE_YIELD.get(crop_type, 4.0)
    
    # 1. کاهش ناشی از شوری
    params = SALINITY_COEFF.get(crop_type, {'threshold': 4.0, 'slope': 0.15})
    threshold = params['threshold']
    slope = params['slope']
    
    if ec <= threshold:
        salinity_reduction = 0
    else:
        salinity_reduction = slope * (ec - threshold)
        salinity_reduction = min(0.95, salinity_reduction)
    
    # 2. کاهش ناشی از pH (اگر داده موجود باشد)
    ph_reduction = 0
    if ph is not None:
        optimal_range = PH_OPTIMAL.get(crop_type, (6.0, 8.0))
        if ph < optimal_range[0]:
            ph_reduction = (optimal_range[0] - ph) * 0.1
        elif ph > optimal_range[1]:
            ph_reduction = (ph - optimal_range[1]) * 0.1
        ph_reduction = min(0.5, max(0, ph_reduction))
    
    # 3. کاهش ناشی از مواد آلی کم
    om_reduction = 0
    if om is not None and om < 1.0:
        om_reduction = (1.0 - om) * 0.15
        om_reduction = min(0.3, om_reduction)
    
    total_reduction = salinity_reduction + ph_reduction + om_reduction
    total_reduction = min(0.95, total_reduction)
    
    actual_yield = base_yield * (1 - total_reduction)
    return round(actual_yield, 2), total_reduction


def calculate_profit(yield_tons_per_ha, crop_type):
    """محاسبه سود خالص و ROI"""
    price = CROP_PRICE.get(crop_type, 10000)
    cost = CROP_COST.get(crop_type, 40000000)
    
    revenue = yield_tons_per_ha * 1000 * price
    profit = revenue - cost
    roi = (profit / cost) * 100 if cost > 0 else 0
    
    return {
        'revenue': int(revenue),
        'profit': int(profit),
        'roi': round(roi, 1),
        'cost': int(cost),
        'price_per_kg': price,
        'yield_tons': yield_tons_per_ha
    }


def rank_crops_by_profit(ec, ph=None, om=None):
    """رتبه‌بندی محصولات بر اساس سودآوری"""
    crops = list(CROP_BASE_YIELD.keys())
    results = []
    
    for crop in crops:
        predicted_yield, reduction = predict_yield(crop, ec, ph, om)
        profit_data = calculate_profit(predicted_yield, crop)
        results.append({
            "crop": crop,
            "crop_fa": get_crop_name_fa(crop),
            "predicted_yield": predicted_yield,
            "yield_reduction_percent": round(reduction * 100, 1),
            "profit": profit_data["profit"],
            "roi": profit_data["roi"],
            "revenue": profit_data["revenue"],
            "cost": profit_data["cost"]
        })
    
    # مرتب‌سازی بر اساس سود
    results.sort(key=lambda x: x["profit"], reverse=True)
    return results


def get_crop_name_fa(crop):
    """تبدیل نام انگلیسی محصول به فارسی"""
    names = {
        'wheat': 'گندم',
        'barley': 'جو',
        'corn': 'ذرت',
        'canola': 'کلزا',
        'sugar_beet': 'چغندر قند',
        'pistachio': 'پسته',
        'saffron': 'زعفران',
        'cotton': 'پنبه',
        'alfalfa': 'یونجه',
        'sunflower': 'آفتابگردان'
    }
    return names.get(crop, crop)

# ============================================================
# بخش ۵: توابع توصیه‌های مدیریتی
# ============================================================

def get_salinity_advice(ec):
    """توصیه بر اساس شوری"""
    if ec < 2:
        return "غیرشور 🌟 - شرایط عالی برای کشت", "success"
    elif ec < 4:
        return "کم‌شور ✅ - مناسب برای بیشتر محصولات", "info"
    elif ec < 8:
        return "شور ⚠️ - استفاده از محصولات مقاوم توصیه می‌شود", "warning"
    else:
        return "بسیار شور ❌ - نیاز به اصلاح جدی دارد", "danger"


def get_ph_advice(ph):
    """توصیه بر اساس اسیدیته"""
    if 6.5 <= ph <= 7.5:
        return "ایده‌آل ✅ - جذب عناصر بهینه است", "success"
    elif 5.5 <= ph < 6.5:
        return "اسیدی ⚠️ - نیاز به آهک دارد (۲ تن در هکتار)", "warning"
    elif 7.5 < ph <= 8.5:
        return "قلیایی ⚠️ - نیاز به گوگرد دارد (۲۰۰ کیلوگرم در هکتار)", "warning"
    else:
        return "بحرانی ❌ - حتماً اصلاح شود", "danger"


def get_om_advice(om):
    """توصیه بر اساس مواد آلی"""
    if om >= 2.5:
        return "عالی 🌟 - خاک غنی از مواد آلی", "success"
    elif om >= 1.5:
        return "خوب ✅ - قابل قبول", "info"
    elif om >= 0.8:
        return "متوسط ⚠️ - نیاز به کود دامی (۳ تن در هکتار)", "warning"
    else:
        return "ضعیف ❌ - نیاز فوری به مواد آلی (۵ تن در هکتار)", "danger"


def get_carbon_advice(carbon):
    """توصیه بر اساس کربن"""
    if carbon > 4:
        return "عالی 🌟 - پتانسیل خوب برای اعتبار کربن", "success"
    elif carbon > 2:
        return "متوسط ⚠️ - قابل بهبود با مدیریت صحیح", "warning"
    else:
        return "ضعیف ❌ - نیاز به افزایش مواد آلی", "danger"

# ============================================================
# بخش ۶: مدل‌های داده Pydantic
# ============================================================

class SoilInput(BaseModel):
    """ورودی‌های مورد نیاز برای پیش‌بینی"""
    ndvi: float = 0.6
    precipitation: float = 600
    temperature: float = 17
    elevation: float = 300
    clay: float = 25
    
    class Config:
        json_schema_extra = {
            "example": {
                "ndvi": 0.6,
                "precipitation": 600,
                "temperature": 17,
                "elevation": 300,
                "clay": 25
            }
        }


class SoilOutput(BaseModel):
    """خروجی‌های پیش‌بینی خصوصیات خاک"""
    ec: float
    ph: float
    organic_matter: float
    texture: str
    carbon: float


class ProfitOutput(BaseModel):
    """خروجی پیش‌بینی سود برای یک محصول"""
    crop: str
    crop_fa: str
    predicted_yield: float
    yield_reduction_percent: float
    profit: int
    roi: float
    revenue: int
    cost: int


class FullAnalysisOutput(BaseModel):
    """تحلیل کامل خاک + سودآوری"""
    soil_properties: Dict[str, Any]
    salinity_advice: str
    ph_advice: str
    om_advice: str
    carbon_advice: str
    ranked_crops: List[ProfitOutput]
    best_crop: ProfitOutput
    water_requirement: int

# ============================================================
# بخش ۷: Endpointهای API
# ============================================================

@app.get("/")
def root():
    """صفحه اصلی API"""
    return {
        "name": "ABAK 2.0 - Unified Soil Analysis API",
        "version": "2.0.0",
        "status": "active",
        "endpoints": {
            "/predict": "POST - پیش‌بینی خصوصیات خاک",
            "/predict-quick": "GET - پیش‌بینی با پارامترهای URL",
            "/full-analysis": "POST - تحلیل کامل با توصیه‌ها",
            "/profit-analysis": "POST - پیش‌بینی سود برای محصولات",
            "/full-analysis-with-profit": "POST - تحلیل کامل + سود",
            "/docs": "Swagger UI - مستندات تعاملی"
        }
    }


@app.post("/predict", response_model=SoilOutput)
def predict(input_data: SoilInput):
    """پیش‌بینی EC, pH, OM, بافت خاک و کربن"""
    result = predict_soil_properties(
        input_data.ndvi, input_data.precipitation,
        input_data.temperature, input_data.elevation, input_data.clay
    )
    return SoilOutput(**result)


@app.get("/predict-quick", response_model=SoilOutput)
def predict_quick(
    ndvi: float = 0.6,
    precipitation: float = 600,
    temperature: float = 17,
    elevation: float = 300,
    clay: float = 25
):
    """پیش‌بینی با پارامترهای URL"""
    result = predict_soil_properties(ndvi, precipitation, temperature, elevation, clay)
    return SoilOutput(**result)


@app.post("/profit-analysis")
def profit_analysis(input_data: SoilInput):
    """پیش‌بینی سود برای محصولات مختلف"""
    soil = predict_soil_properties(
        input_data.ndvi, input_data.precipitation,
        input_data.temperature, input_data.elevation, input_data.clay
    )
    
    ranked = rank_crops_by_profit(soil['ec'], soil['ph'], soil['organic_matter'])
    
    return {
        "soil_ec": soil['ec'],
        "soil_ph": soil['ph'],
        "soil_om": soil['organic_matter'],
        "ranked_crops": ranked[:5],
        "best_crop": ranked[0]
    }


@app.post("/full-analysis-with-profit", response_model=FullAnalysisOutput)
def full_analysis_with_profit(input_data: SoilInput):
    """تحلیل کامل خاک + پیش‌بینی سود برای محصولات مختلف"""
    
    # مرحله ۱: پیش‌بینی خصوصیات خاک
    soil = predict_soil_properties(
        input_data.ndvi, input_data.precipitation,
        input_data.temperature, input_data.elevation, input_data.clay
    )
    
    # مرحله ۲: توصیه‌ها
    salinity_text, salinity_status = get_salinity_advice(soil['ec'])
    ph_text, ph_status = get_ph_advice(soil['ph'])
    om_text, om_status = get_om_advice(soil['organic_matter'])
    carbon_text, carbon_status = get_carbon_advice(soil['carbon'])
    
    # مرحله ۳: رتبه‌بندی محصولات بر اساس سودآوری
    ranked = rank_crops_by_profit(soil['ec'], soil['ph'], soil['organic_matter'])
    
    # مرحله ۴: نیاز آبی بهترین محصول
    best_crop_name = ranked[0]['crop']
    water_req = WATER_REQUIREMENT.get(best_crop_name, 5000)
    
    return FullAnalysisOutput(
        soil_properties=soil,
        salinity_advice=salinity_text,
        ph_advice=ph_text,
        om_advice=om_text,
        carbon_advice=carbon_text,
        ranked_crops=[ProfitOutput(**crop) for crop in ranked[:5]],
        best_crop=ProfitOutput(**ranked[0]),
        water_requirement=water_req
    )


@app.get("/health")
def health_check():
    """بررسی سلامت API"""
    return {
        "status": "healthy",
        "models_loaded": model_ec is not None,
        "model_ec": "loaded" if model_ec else "not loaded",
        "model_ph": "loaded" if model_ph else "not loaded",
        "model_om": "loaded" if model_om else "not loaded",
        "model_texture": "loaded" if model_texture else "not loaded",
        "crops_supported": len(CROP_BASE_YIELD)
    }


# ============================================================
# بخش ۸: راه‌اندازی سرور
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 ABAK 2.0 Unified API - نسخه نهایی")
    print("=" * 70)
    print("📊 پیش‌بینی: EC, pH, مواد آلی, بافت خاک, کربن")
    print("💰 پیش‌بینی سود و ROI برای ۱۰ محصول")
    print("🌐 مستندات Swagger: http://localhost:8001/docs")
    print("🔍 تست سریع: http://localhost:8001/predict-quick?ndvi=0.6&precipitation=600&temperature=17&elevation=300&clay=25")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)