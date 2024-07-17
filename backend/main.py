import asyncio
from hypercorn.config import Config
from hypercorn.asyncio import serve
from app import create_app

app = create_app()

async def main():
    config = Config()
    config.bind = ["0.0.0.0:5000"]
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())