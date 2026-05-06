# raster_maker.py
# SoilStress Vision - روز ۳
# تبدیل نتایج نقاط به داده‌های قابل نمایش روی نقشه

import json
import numpy as np
from scipy.interpolate import griddata
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# ============================================================
# تبدیل نقاط به شبکه منظم (Grid)
# ============================================================

def points_to_grid(coords: List[Tuple[float, float]], 
                   values: List[float],
                   grid_size: Tuple[int, int] = (100, 100),
                   method: str = 'linear') -> Dict:
    """
    تبدیل نقاط پراکنده به یک شبکه منظم (ایدیال برای نقشه)
    
    Parameters:
    -----------
    coords : list of tuples
        لیست مختصات (lat, lon)
    values : list of float
        لیست مقادیر متناظر
    grid_size : tuple
        ابعاد شبکه خروجی (rows, cols)
    method : str
        روش درون‌یابی ('linear', 'nearest', 'cubic')
    
    Returns:
    --------
    grid_data : dict
        شامل grid_x, grid_y, grid_z و محدوده‌ها
    """
    
    if len(coords) < 4:
        print("⚠️ تعداد نقاط برای درون‌یابی کافی نیست. از روش نزدیک‌ترین همسایه استفاده می‌شود.")
        method = 'nearest'
    
    # تبدیل به آرایه numpy
    points = np.array(coords)
    values = np.array(values)
    
    # پیدا کردن محدوده جغرافیایی
    lat_min, lat_max = points[:, 0].min(), points[:, 0].max()
    lon_min, lon_max = points[:, 1].min(), points[:, 1].max()
    
    # ایجاد شبکه منظم
    lat_grid = np.linspace(lat_min, lat_max, grid_size[0])
    lon_grid = np.linspace(lon_min, lon_max, grid_size[1])
    grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)
    
    # درون‌یابی
    grid_z = griddata(points, values, (grid_lat, grid_lon), method=method)
    
    # پر کردن نقاط NaN (با نزدیک‌ترین مقدار)
    if np.any(np.isnan(grid_z)):
        grid_z_filled = griddata(points, values, (grid_lat, grid_lon), method='nearest')
        grid_z = np.where(np.isnan(grid_z), grid_z_filled, grid_z)
    
    return {
        "lat_grid": lat_grid.tolist(),
        "lon_grid": lon_grid.tolist(),
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "grid_z": grid_z.tolist(),
        "grid_z_min": float(np.nanmin(grid_z)),
        "grid_z_max": float(np.nanmax(grid_z))
    }


# ============================================================
# تولید رنگ‌بندی برای نقشه
# ============================================================

def get_color_for_value(value, vmin, vmax, parameter='ec'):
    """
    تبدیل مقدار به رنگ HEX بر اساس پارامتر
    """
    # نرمال‌سازی (0 تا 1)
    if vmax > vmin:
        norm = (value - vmin) / (vmax - vmin)
    else:
        norm = 0.5
    
    norm = max(0, min(1, norm))
    
    # رنگ‌بندی سبز ← زرد ← قرمز
    if parameter in ['ec', 'salinity']:
        # شوری: سبز (کم) → زرد → قرمز (زیاد)
        if norm < 0.5:
            # سبز به زرد
            r = int(255 * (norm * 2))
            g = 255
            b = 0
        else:
            # زرد به قرمز
            r = 255
            g = int(255 * (1 - (norm - 0.5) * 2))
            b = 0
    elif parameter in ['carbon', 'organic_matter']:
        # کربن: قرمز (کم) → زرد → سبز (زیاد) (معکوس شوری)
        if norm < 0.5:
            r = 255
            g = int(255 * (norm * 2))
            b = 0
        else:
            r = int(255 * (1 - (norm - 0.5) * 2))
            g = 255
            b = 0
    elif parameter == 'ph':
        # pH: قرمز (اسیدی/قلیایی) → سبز (خنثی)
        # ایده‌آل 6.5-7.5
        if 6.5 <= value <= 7.5:
            norm = 0.5
        elif value < 6.5:
            norm = max(0, (6.5 - value) / 3.5)
        else:
            norm = min(1, (value - 7.5) / 2.5)
        
        if norm < 0.5:
            r = 255
            g = int(255 * (1 - norm * 2))
            b = 0
        else:
            r = int(255 * ((norm - 0.5) * 2))
            g = 255
            b = 0
    else:
        # پیش‌فرض (سفید تا سبز)
        r = int(255 * (1 - norm))
        g = 255
        b = int(255 * (1 - norm))
    
    return f"rgb({r}, {g}, {b})"


def generate_color_map(vmin, vmax, parameter='ec'):
    """
    تولید نقشه رنگی برای نمایش روی Leaflet
    """
    colors = []
    for i in range(10):
        value = vmin + (i / 9) * (vmax - vmin)
        colors.append(get_color_for_value(value, vmin, vmax, parameter))
    
    return colors


# ============================================================
# خروجی برای Leaflet
# ============================================================

def create_leaflet_layer(grid_data: Dict, parameter: str = 'ec') -> Dict:
    """
    تبدیل داده‌های grid به فرمت قابل نمایش در Leaflet
    """
    # تبدیل grid_z به لیست رنگ‌ها
    z_min = grid_data["grid_z_min"]
    z_max = grid_data["grid_z_max"]
    
    colors_2d = []
    for i, row in enumerate(grid_data["grid_z"]):
        color_row = []
        for j, value in enumerate(row):
            if value is not None and not np.isnan(value):
                color = get_color_for_value(value, z_min, z_max, parameter)
                color_row.append(color)
            else:
                color_row.append("rgb(200,200,200)")
        colors_2d.append(color_row)
    
    return {
        "lat_min": grid_data["lat_min"],
        "lat_max": grid_data["lat_max"],
        "lon_min": grid_data["lon_min"],
        "lon_max": grid_data["lon_max"],
        "colors": colors_2d,
        "z_min": round(z_min, 2),
        "z_max": round(z_max, 2),
        "parameter": parameter
    }


# ============================================================
# تست روز ۳
# ============================================================

if __name__ == "__main__":
    from batch_api import analyze_points_batch, points_to_raster_data
    
    print("=" * 60)
    print("SoilStress Vision - تست Raster Maker (روز ۳)")
    print("=" * 60)
    
    # نقاط تست (همان ۵ نقطه قبلی)
    test_points = [
        (35.6892, 51.3890),  # تهران
        (36.2605, 59.6168),  # مشهد
        (32.6546, 51.6680),  # اصفهان
        (29.5918, 52.5837),  # شیراز
        (31.3183, 48.6706),  # اهواز
    ]
    
    print("\n📊 تحلیل نقاط...")
    results = analyze_points_batch(test_points, max_workers=5)
    
    # تبدیل به داده رستر
    raster_data = points_to_raster_data(results, parameter="ec")
    
    if len(raster_data["coords"]) > 0:
        print(f"\n🗺️ تعداد نقاط معتبر: {len(raster_data['coords'])}")
        print(f"   محدوده شوری: {raster_data['vmin']:.2f} - {raster_data['vmax']:.2f} dS/m")
        
        # تبدیل به شبکه منظم
        grid_data = points_to_grid(raster_data["coords"], raster_data["values"], grid_size=(50, 50))
        print(f"\n📐 شبکه تولید شده: {len(grid_data['lat_grid'])} x {len(grid_data['lon_grid'])}")
        print(f"   محدوده عرض: {grid_data['lat_min']:.4f} - {grid_data['lat_max']:.4f}")
        print(f"   محدوده طول: {grid_data['lon_min']:.4f} - {grid_data['lon_max']:.4f}")
        print(f"   محدوده شوری در شبکه: {grid_data['grid_z_min']:.2f} - {grid_data['grid_z_max']:.2f}")
        
        # تولید لایه Leaflet
        leaflet_layer = create_leaflet_layer(grid_data, parameter="ec")
        print(f"\n🎨 لایه Leaflet ساخته شد:")
        print(f"   تعداد ردیف‌های رنگ: {len(leaflet_layer['colors'])}")
        print(f"   محدوده رنگ: {leaflet_layer['z_min']} - {leaflet_layer['z_max']}")
        
        # ذخیره خروجی برای استفاده در فرانت‌اند
        output_file = "../frontend/soil_data.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(leaflet_layer, f, ensure_ascii=False, indent=2)
        print(f"\n💾 خروجی در {output_file} ذخیره شد")
    else:
        print("❌ هیچ داده معتبری یافت نشد")
    
    print("\n" + "=" * 60)
    print("✅ روز ۳ تمام شد. raster_maker.py کار می‌کند.")
    print("=" * 60)