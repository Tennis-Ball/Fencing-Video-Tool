# Load configuration
import config

# Load flask imports
from flask import Flask


# Setup main flask application
app = Flask(__name__)
app.config.from_object(config.TestConfig)

# Setup SQL alchemy
# db = SQLAlchemy(app)
#
# # migration info: https://flask-migrate.readthedocs.io/en/latest/
# migrate = Migrate(app, db)  # use $ flask db migrate -m "COMMIT NAME" then $ flask db upgrade for updating any tables
#
# # Login manager setup
# login = LoginManager(app)
# login.login_view = 'login'

# Load routes
from app import routes


# Route error handlers to their pages
# app.register_error_handler(404, routes.page_not_found)
