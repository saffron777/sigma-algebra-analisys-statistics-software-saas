"""Flask blueprints: thin controllers that validate input, call one use case and serialise the output."""

from flask import Flask

from sigma.presentation.web.blueprints import (
    ai_api,
    analysis_api,
    anova_api,
    assistant_api,
    auth_api,
    auth_pages,
    console_api,
    database_api,
    datasets_api,
    descriptive_api,
    distributions_api,
    export_api,
    hypothesis_api,
    markets_api,
    pages,
    regression_api,
    results_api,
    templates_api,
    users_api,
    workspace_api,
)

ALL_BLUEPRINTS = (
    pages.bp,
    auth_pages.bp,
    auth_api.bp,
    users_api.bp,
    datasets_api.bp,
    distributions_api.bp,
    descriptive_api.bp,
    hypothesis_api.bp,
    regression_api.bp,
    anova_api.bp,
    markets_api.bp,
    assistant_api.bp,
    results_api.bp,
    console_api.bp,
    ai_api.bp,
    ai_api.admin_bp,
    analysis_api.bp,
    database_api.admin_bp,
    database_api.setup_bp,
    workspace_api.bp,
    export_api.bp,
    templates_api.bp,
)


def register_blueprints(app: Flask) -> None:
    for blueprint in ALL_BLUEPRINTS:
        app.register_blueprint(blueprint)
