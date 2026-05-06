# batch_api.py
# SoilStress Vision - روز ۲
# فراخوانی همزمان ABAK API برای هزاران نقطه

import requests
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import numpy as np

# تنظیمات API
API_URL = "http://127.0.0.1:8001/predict-quick"
CACHE_FILE = "data/api_cache.json"

# ============================================================
# مدیریت کش (Cache) برای جلوگیری از درخواست‌های تکراری
# ============================================================

def load_cache():
    """بارگذاری کش از فایل"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """ذخیره کش در فایل"""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def get_cache_key(ndvi, precip, temp, elev, clay):
    """ایجاد کلید یکتا برای هر درخواست"""
    return f"{ndvi:.3f}_{precip:.1f}_{temp:.1f}_{elev:.1f}_{clay:.1f}"

# ============================================================
# تخمین داده‌های محیطی از روی مختصات
# ============================================================

def estimate_environmental_data(lat: float, lon: float) -> Dict:
    """
    تخمین NDVI، بارش، دما، ارتفاع و درصد رس از روی مختصات
    این یک تخمین ساده است. در نسخه واقعی باید از APIهای معتبر استفاده کرد.
    """
    # NDVI: شمال ایران سبزتر
    ndvi = 0.4 + (37 - lat) / 35
    ndvi = max(0.15, min(0.85, ndvi))
    
    # بارش: شمال بیشتر، جنوب کمتر
    precip = 150 + (37 - lat) * 45
    precip = max(100, min(900, precip))
    
    # دما: جنوب گرمتر
    temp = 12 + (35 - lat) * 0.45
    temp = max(8, min(28, temp))
    
    # ارتفاع: تخمین ساده
    elev = 200 + abs(35 - lat) * 100
    elev = max(50, min(2500, elev))
    
    # درصد رس: تخمین ساده
    clay = 15 + abs(lat - 32) * 1.4
    clay = max(10, min(50, clay))
    
    return {
        "ndvi": round(ndvi, 4),
        "precipitation": round(precip, 1),
        "temperature": round(temp, 1),
        "elevation": round(elev, 1),
        "clay": round(clay, 1)
    }


def call_abaK_api(ndvi, precip, temp, elev, clay, timeout=10):
    """
    فراخوانی ABAK API برای یک نقطه
    """
    params = {
        "ndvi": ndvi,
        "precipitation": precip,
        "temperature": temp,
        "elevation": elev,
        "clay": clay
    }
    
    try:
        response = requests.get(API_URL, params=params, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def analyze_point(lat: float, lon: float, use_cache: bool = True) -> Dict:
    """
    تحلیل یک نقطه کامل (تخمین داده + فراخوانی API + کش)
    """
    # تخمین داده‌های محیطی
    env_data = estimate_environmental_data(lat, lon)
    
    # بررسی کش
    cache = load_cache() if use_cache else {}
    cache_key = get_cache_key(
        env_data["ndvi"], env_data["precipitation"],
        env_data["temperature"], env_data["elevation"], env_data["clay"]
    )
    
    if use_cache and cache_key in cache:
        result = cache[cache_key]
        result["from_cache"] = True
        return {
            "lat": lat,
            "lon": lon,
            "env": env_data,
            "soil": result
        }
    
    # فراخوانی API
    api_result = call_abaK_api(
        env_data["ndvi"], env_data["precipitation"],
        env_data["temperature"], env_data["elevation"], env_data["clay"]
    )
    
    # ذخیره در کش
    if use_cache and "error" not in api_result:
        cache[cache_key] = api_result
        save_cache(cache)
    
    return {
        "lat": lat,
        "lon": lon,
        "env": env_data,
        "soil": api_result
    }


# ============================================================
# پردازش دسته‌ای (Batch Processing)
# ============================================================

def analyze_points_batch(points: List[Tuple[float, float]], 
                         max_workers: int = 10,
                         use_cache: bool = True,
                         on_progress=None) -> List[Dict]:
    """
    تحلیل همزمان چندین نقطه با ThreadPool
    
    Parameters:
    -----------
    points : list of tuples
        لیست نقاط (lat, lon)
    max_workers : int
        تعداد تردهای همزمان
    use_cache : bool
        استفاده از کش یا خیر
    on_progress : callable
        تابع پیشرفت (اختیاری)
    
    Returns:
    --------
    results : list of dict
        نتایج تحلیل برای هر نقطه
    """
    results = []
    total = len(points)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_point = {
            executor.submit(analyze_point, lat, lon, use_cache): (lat, lon)
            for lat, lon in points
        }
        
        for future in as_completed(future_to_point):
            result = future.result()
            results.append(result)
            completed += 1
            
            if on_progress:
                on_progress(completed, total)
    
    return results


def points_to_raster_data(results: List[Dict], parameter: str = "ec") -> Dict:
    """
    تبدیل نتایج نقاط به داده‌های قابل نمایش روی نقشه
    
    Parameters:
    -----------
    results : list of dict
        خروجی analyze_points_batch
    parameter : str
        پارامتر مورد نظر ("ec", "ph", "organic_matter", "carbon")
    
    Returns:
    --------
    raster_data : dict
        شامل لیست مختصات و مقادیر
    """
    coords = []
    values = []
    
    for r in results:
        if "soil" in r and parameter in r["soil"]:
            coords.append((r["lat"], r["lon"]))
            values.append(r["soil"][parameter])
        elif "soil" in r and "error" in r["soil"]:
            # در صورت خطا، مقدار None
            coords.append((r["lat"], r["lon"]))
            values.append(None)
    
    # نرمال‌سازی مقادیر برای رنگ‌بندی
    valid_values = [v for v in values if v is not None]
    if valid_values:
        vmin = min(valid_values)
        vmax = max(valid_values)
    else:
        vmin, vmax = 0, 1
    
    return {
        "coords": coords,
        "values": values,
        "vmin": vmin,
        "vmax": vmax,
        "parameter": parameter
    }


# ============================================================
# تست روز ۲
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SoilStress Vision - تست Batch API (روز ۲)")
    print("=" * 60)
    
    # تست با چند نقطه نمونه
    test_points = [
        (35.6892, 51.3890),  # تهران
        (36.2605, 59.6168),  # مشهد
        (32.6546, 51.6680),  # اصفهان
        (29.5918, 52.5837),  # شیراز
        (31.3183, 48.6706),  # اهواز
    ]
    
    print(f"\n📊 تست با {len(test_points)} نقطه")
    
    # تابع پیشرفت
    def show_progress(completed, total):
        print(f"   پیشرفت: {completed}/{total} نقاط تحلیل شد")
    
    # تحلیل نقاط
    results = analyze_points_batch(test_points, max_workers=5, on_progress=show_progress)
    
    # نمایش نتایج
    print("\n📈 نتایج تحلیل:")
    for i, r in enumerate(results):
        print(f"\n   نقطه {i+1}: ({r['lat']:.4f}, {r['lon']:.4f})")
        if "soil" in r and "ec" in r["soil"]:
            print(f"      شوری (EC): {r['soil']['ec']} dS/m")
            print(f"      pH: {r['soil']['ph']}")
            print(f"      مواد آلی: {r['soil']['organic_matter']}%")
            print(f"      کربن: {r['soil']['carbon']}%")
        else:
            print(f"      خطا: {r['soil'].get('error', 'نامشخص')}")
    
    # تبدیل به داده رستر
    print("\n🗺️ تبدیل به داده رستر (شوری):")
    raster_data = points_to_raster_data(results, parameter="ec")
    print(f"   تعداد نقاط معتبر: {len([v for v in raster_data['values'] if v is not None])}")
    print(f"   محدوده شوری: {raster_data['vmin']:.2f} - {raster_data['vmax']:.2f} dS/m")
    
