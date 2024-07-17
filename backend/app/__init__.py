from flask import Flask
from flask_cors import CORS
from .routes import main as main_blueprint
from .query_operation import Neo4jOperations
import logging

logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    CORS(app)

    app.neo4j_ops = Neo4jOperations()
    
    @app.before_request
    async def initialize_neo4j():
        if not app.neo4j_ops.is_connected:
            logger.info("Initializing Neo4j connection...")
            await app.neo4j_ops.connect()

    @app.teardown_appcontext
    async def close_neo4j_driver(exception=None):
        if hasattr(app, 'neo4j_ops') and app.neo4j_ops.is_connected:
            await app.neo4j_ops.close()

    app.register_blueprint(main_blueprint)

    return app