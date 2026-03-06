"""
FastAPI endpoint for serving generated charts.
"""
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from fastapi.responses import FileResponse

from factorybench.api.app import app
from factorybench.viz.charts import generate_all_charts
from factorybench.config import RUN_DIR


CHARTS_DIR = Path("charts")


@app.get("/charts/{chart_type}")
async def get_chart(
    chart_type: str,
    stage: Optional[str] = None,
    regenerate: bool = False
):
    """
    Get a generated chart image.
    
    Args:
        chart_type: Type of chart (error_distribution, convergence, etc.)
        stage: Optional stage filter
        regenerate: Force regeneration of charts
        
    Returns:
        Chart image file
    """
    if regenerate or not CHARTS_DIR.exists():
        # Generate charts
        generate_all_charts(RUN_DIR, CHARTS_DIR)
    
    chart_file = CHARTS_DIR / f"{chart_type}.png"
    
    if not chart_file.exists():
        raise HTTPException(status_code=404, detail=f"Chart {chart_type} not found")
    
    return FileResponse(chart_file, media_type="image/png")


@app.post("/charts/regenerate")
async def regenerate_charts():
    """Regenerate all charts from current runs."""
    generate_all_charts(RUN_DIR, CHARTS_DIR)
    return {"status": "ok", "charts_dir": str(CHARTS_DIR)}
