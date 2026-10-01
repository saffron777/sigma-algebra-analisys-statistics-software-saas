"""Flask blueprints: thin controllers that validate input, call one use case and serialise the output."""

from flask import Flask

from sigma.presentation.web.blueprints import (
    anova_api,
    auth_api,
    auth_pages,
    console_api,
    datasets_api,
    descriptive_api,
    distributions_api,
    hypothesis_api,
    markets_api,
    pages,
    regression_api,
    results_api,
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
    results_api.bp,
    console_api.bp,
    workspace_api.bp,
)


def register_blueprints(app: Flask) -> None:
    for blueprint in ALL_BLUEPRINTS:
        app.register_blueprint(blueprint)
