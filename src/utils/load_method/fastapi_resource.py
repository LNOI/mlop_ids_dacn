from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from src.utils.load_method.load_utils import register_load_method


@register_load_method
def fastapi_app() -> FastAPI:
    app = FastAPI(
        title="FGD",
        description="FastAPI server for Fashion Garment Detector.",
        version="1.0.0"
    )

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title="FGD",
            version="1.0.0",
            description="FastAPI server for Fashion Garment Detector.",
            routes=app.routes,
        )
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    app.openapi = custom_openapi

    return app
