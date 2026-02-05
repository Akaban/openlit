"""
Utilities for Qdrant instrumentation.
"""

import time
from urllib.parse import urlparse
from opentelemetry.trace import Status, StatusCode
from openlit.__helpers import (
    common_db_span_attributes,
    record_db_metrics,
)
from openlit.semcov import SemanticConvention

# Operation mapping for Qdrant endpoints
DB_OPERATION_MAP = {
    "qdrant.create_collection": SemanticConvention.DB_OPERATION_CREATE_COLLECTION,
    "qdrant.upload_collection": SemanticConvention.DB_OPERATION_CREATE_COLLECTION,
    "qdrant.delete_collection": SemanticConvention.DB_OPERATION_DELETE_COLLECTION,
    "qdrant.update_collection": SemanticConvention.DB_OPERATION_UPDATE_COLLECTION,
    "qdrant.upsert": SemanticConvention.DB_OPERATION_UPSERT,
    "qdrant.upload_points": SemanticConvention.DB_OPERATION_INSERT,
    "qdrant.set_payload": SemanticConvention.DB_OPERATION_INSERT,
    "qdrant.overwrite_payload": SemanticConvention.DB_OPERATION_UPDATE,
    "qdrant.update_vectors": SemanticConvention.DB_OPERATION_UPDATE,
    "qdrant.delete": SemanticConvention.DB_OPERATION_DELETE,
    "qdrant.delete_vectors": SemanticConvention.DB_OPERATION_DELETE,
    "qdrant.delete_payload": SemanticConvention.DB_OPERATION_DELETE,
    "qdrant.clear_payload": SemanticConvention.DB_OPERATION_DELETE,
    "qdrant.retrieve": SemanticConvention.DB_OPERATION_GET,
    "qdrant.scroll": SemanticConvention.DB_OPERATION_GET,
    "qdrant.query": SemanticConvention.DB_OPERATION_GET,
    "qdrant.query_batch": SemanticConvention.DB_OPERATION_GET,
    "qdrant.query_points": SemanticConvention.DB_OPERATION_GET,
    "qdrant.query_batch_points": SemanticConvention.DB_OPERATION_GET,
    "qdrant.query_points_groups": SemanticConvention.DB_OPERATION_GET,
    "qdrant.create_payload_index": SemanticConvention.DB_OPERATION_CREATE_INDEX,
}


def object_count(obj):
    """
    Counts Length of object if it exists, Else returns 0.
    """
    try:
        return len(obj)
    except (TypeError, AttributeError):
        return 0


def format_query_response(response):
    """
    Format query response as JSON string of points.
    """
    import json

    if response is None:
        return ""

    # Handle QueryResponse (has .points)
    points = getattr(response, "points", None)
    if points is None:
        if isinstance(response, list):
            points = response
        else:
            return ""

    if not points:
        return "[]"

    # Convert points to serializable format
    points_data = []
    for point in points:
        point_dict = {}
        if hasattr(point, "id"):
            point_dict["id"] = point.id
        if hasattr(point, "score"):
            point_dict["score"] = point.score
        if hasattr(point, "payload"):
            point_dict["payload"] = point.payload
        if hasattr(point, "vector"):
            # Skip vectors - too large
            pass
        points_data.append(point_dict)

    return json.dumps(points_data, default=str)


def format_query_points_ast(query, prefetch):
    """
    Format query_points call as AST-like representation.
    Returns (query_text, filters_list).

    Example output: "RRF([Prefetch(using=gemini-embedding-001, limit=150), Prefetch(using=bm25, limit=150)])"
    """
    filters = []

    # Get fusion type from query
    fusion_type = None
    if hasattr(query, "fusion"):
        fusion_obj = query.fusion
        fusion_type = fusion_obj.name if hasattr(fusion_obj, "name") else str(fusion_obj).split(".")[-1].rstrip("'>")
    elif isinstance(query, dict) and "fusion" in query:
        fusion_type = str(query["fusion"])

    if not prefetch:
        return str(query)[:500], filters

    prefetch_list = prefetch if isinstance(prefetch, list) else [prefetch]
    prefetch_parts = []

    for pf in prefetch_list:
        # Extract attributes (handle both objects and dicts)
        if hasattr(pf, "using"):
            pf_using = pf.using
            pf_limit = pf.limit
            pf_filter = getattr(pf, "filter", None)
        elif isinstance(pf, dict):
            pf_using = pf.get("using")
            pf_limit = pf.get("limit")
            pf_filter = pf.get("filter")
        else:
            continue

        # Build prefetch representation
        pf_args = []
        if pf_using:
            pf_args.append(f"using={pf_using}")
        if pf_limit:
            pf_args.append(f"limit={pf_limit}")
        if pf_filter:
            filters.append(str(pf_filter))
            pf_args.append("filter=...")

        prefetch_parts.append(f"Prefetch({', '.join(pf_args)})")

    # Format as AST: RRF([Prefetch(...), Prefetch(...)])
    if fusion_type:
        query_text = f"{fusion_type}([{', '.join(prefetch_parts)}])"
    else:
        query_text = f"[{', '.join(prefetch_parts)}]"

    return query_text[:500], filters


def set_server_address_and_port(instance):
    """
    Extracts server address and port from Qdrant client instance.

    Args:
        instance: Qdrant client instance

    Returns:
        tuple: (server_address, server_port)
    """
    server_address = "localhost"
    server_port = 6333

    # Try to extract actual server info from Qdrant init_options
    if hasattr(instance, "init_options") and instance.init_options:
        init_options = instance.init_options

        # Get URL and extract host/port from it
        url = init_options.get("url", "")
        if url:
            try:
                parsed = urlparse(url)
                if parsed.hostname:
                    server_address = parsed.hostname
                if parsed.port:
                    server_port = parsed.port
            except Exception:
                pass

        # Also try direct port from init_options if URL parsing didnt work
        if "port" in init_options and init_options["port"]:
            server_port = init_options["port"]

    return server_address, server_port


def common_qdrant_logic(
    scope,
    environment,
    application_name,
    metrics,
    capture_message_content,
    disable_metrics,
    version,
    instance=None,
    endpoint=None,
):
    """
    Process Qdrant request and generate telemetry.
    """
    scope._end_time = time.time()

    # Set common database span attributes using helper
    common_db_span_attributes(
        scope,
        SemanticConvention.DB_SYSTEM_QDRANT,
        scope._server_address,
        scope._server_port,
        environment,
        application_name,
        version,
    )

    # Set DB operation specific attributes
    scope._span.set_attribute(SemanticConvention.DB_OPERATION_NAME, scope._db_operation)
    scope._span.set_attribute(
        SemanticConvention.DB_CLIENT_OPERATION_DURATION,
        scope._end_time - scope._start_time,
    )

    # Handle collection management operations
    if scope._db_operation in [
        SemanticConvention.DB_OPERATION_CREATE_COLLECTION,
        SemanticConvention.DB_OPERATION_DELETE_COLLECTION,
        SemanticConvention.DB_OPERATION_UPDATE_COLLECTION,
    ]:
        collection_name = scope._kwargs.get("collection_name", "unknown")

        scope._span.set_attribute(
            SemanticConvention.DB_COLLECTION_NAME, collection_name
        )
        scope._span.set_attribute(
            SemanticConvention.DB_QUERY_TEXT, f"Collection: {collection_name}"
        )
        scope._span.set_attribute(
            SemanticConvention.DB_QUERY_SUMMARY,
            f"{scope._db_operation} {collection_name}",
        )

    # Handle data insertion operations
    elif scope._db_operation in [
        SemanticConvention.DB_OPERATION_INSERT,
        SemanticConvention.DB_OPERATION_UPSERT,
    ]:
        collection_name = scope._kwargs.get("collection_name", "unknown")

        if endpoint == "qdrant.set_payload":
            points = scope._kwargs.get("points", [])
            payload = scope._kwargs.get("payload", {})

            scope._span.set_attribute(
                SemanticConvention.DB_COLLECTION_NAME, collection_name
            )
            scope._span.set_attribute(SemanticConvention.DB_QUERY_TEXT, str(points))
            scope._span.set_attribute(
                SemanticConvention.DB_VECTOR_COUNT, object_count(points)
            )
            scope._span.set_attribute(
                SemanticConvention.DB_PAYLOAD_COUNT, object_count(payload)
            )

            # Set operation status if response available
            if scope._response and hasattr(scope._response, "status"):
                scope._span.set_attribute(
                    SemanticConvention.DB_OPERATION_STATUS, scope._response.status
                )

            scope._span.set_attribute(
                SemanticConvention.DB_QUERY_SUMMARY,
                f"{scope._db_operation} {collection_name} "
                f"points={object_count(points)} "
                f"payload={object_count(payload)}",
            )

        elif endpoint in ["qdrant.upsert", "qdrant.upload_points"]:
            points = scope._kwargs.get("points", [])

            scope._span.set_attribute(
                SemanticConvention.DB_COLLECTION_NAME, collection_name
            )
            scope._span.set_attribute(SemanticConvention.DB_QUERY_TEXT, str(points))
            scope._span.set_attribute(
                SemanticConvention.DB_VECTOR_COUNT, object_count(points)
            )

            # Set operation status if response available
            if scope._response and hasattr(scope._response, "status"):
                scope._span.set_attribute(
                    SemanticConvention.DB_OPERATION_STATUS, scope._response.status
                )

            scope._span.set_attribute(
                SemanticConvention.DB_QUERY_SUMMARY,
                f"{scope._db_operation} {collection_name} "
                f"points={object_count(points)}",
            )

    # Handle data update operations
    elif scope._db_operation == SemanticConvention.DB_OPERATION_UPDATE:
        collection_name = scope._kwargs.get("collection_name", "unknown")

        if endpoint == "qdrant.overwrite_payload":
            points = scope._kwargs.get("points", [])
            payload = scope._kwargs.get("payload", {})

            scope._span.set_attribute(
                SemanticConvention.DB_COLLECTION_NAME, collection_name
            )
            scope._span.set_attribute(SemanticConvention.DB_QUERY_TEXT, str(points))
            scope._span.set_attribute(
                SemanticConvention.DB_VECTOR_COUNT, object_count(points)
            )
            scope._span.set_attribute(
                SemanticConvention.DB_PAYLOAD_COUNT, object_count(payload)
            )

            # Set operation status if response available
            if scope._response and hasattr(scope._response, "status"):
                scope._span.set_attribute(
                    SemanticConvention.DB_OPERATION_STATUS, scope._response.status
                )

            scope._span.set_attribute(
                SemanticConvention.DB_QUERY_SUMMARY,
                f"{scope._db_operation} {collection_name} "
                f"points={object_count(points)} "
                f"payload={object_count(payload)}",
            )

        elif endpoint == "qdrant.update_vectors":
            points = scope._kwargs.get("points", [])

            scope._span.set_attribute(
                SemanticConvention.DB_COLLECTION_NAME, collection_name
            )
            scope._span.set_attribute(SemanticConvention.DB_QUERY_TEXT, str(points))
            scope._span.set_attribute(
                SemanticConvention.DB_VECTOR_COUNT, object_count(points)
            )

            # Set operation status if response available
            if scope._response and hasattr(scope._response, "status"):
                scope._span.set_attribute(
                    SemanticConvention.DB_OPERATION_STATUS, scope._response.status
                )

            scope._span.set_attribute(
                SemanticConvention.DB_QUERY_SUMMARY,
                f"{scope._db_operation} {collection_name} "
                f"points={object_count(points)}",
            )

    # Handle data deletion operations
    elif scope._db_operation == SemanticConvention.DB_OPERATION_DELETE:
        collection_name = scope._kwargs.get("collection_name", "unknown")

        if endpoint in ["qdrant.delete_payload", "qdrant.delete_vectors"]:
            points = scope._kwargs.get("points", [])

            scope._span.set_attribute(
                SemanticConvention.DB_COLLECTION_NAME, collection_name
            )
            scope._span.set_attribute(SemanticConvention.DB_QUERY_TEXT, str(points))
            scope._span.set_attribute(
                SemanticConvention.DB_VECTOR_COUNT, object_count(points)
            )

            # Set operation status if response available
            if scope._response and hasattr(scope._response, "status"):
                scope._span.set_attribute(
                    SemanticConvention.DB_OPERATION_STATUS, scope._response.status
                )

            scope._span.set_attribute(
                SemanticConvention.DB_QUERY_SUMMARY,
                f"{scope._db_operation} {collection_name} "
                f"points={object_count(points)}",
            )

        elif endpoint in ["qdrant.clear_payload", "qdrant.delete"]:
            points_selector = scope._kwargs.get("points_selector", [])

            scope._span.set_attribute(
                SemanticConvention.DB_COLLECTION_NAME, collection_name
            )
            scope._span.set_attribute(
                SemanticConvention.DB_QUERY_TEXT, str(points_selector)
            )
            scope._span.set_attribute(
                SemanticConvention.DB_VECTOR_COUNT, object_count(points_selector)
            )

            # Set operation status if response available
            if scope._response and hasattr(scope._response, "status"):
                scope._span.set_attribute(
                    SemanticConvention.DB_OPERATION_STATUS, scope._response.status
                )

            scope._span.set_attribute(
                SemanticConvention.DB_QUERY_SUMMARY,
                f"{scope._db_operation} {collection_name} "
                f"selector={object_count(points_selector)}",
            )

    # Handle query operations
    elif scope._db_operation == SemanticConvention.DB_OPERATION_GET:
        collection_name = scope._kwargs.get("collection_name", "unknown")

        if endpoint == "qdrant.retrieve":
            ids = scope._kwargs.get("ids", [])

            scope._span.set_attribute(
                SemanticConvention.DB_COLLECTION_NAME, collection_name
            )
            scope._span.set_attribute(SemanticConvention.DB_QUERY_TEXT, str(ids))
            scope._span.set_attribute(
                SemanticConvention.DB_VECTOR_COUNT, object_count(ids)
            )

            scope._span.set_attribute(
                SemanticConvention.DB_QUERY_SUMMARY,
                f"{scope._db_operation} {collection_name} ids={ids}",
            )

        elif endpoint == "qdrant.scroll":
            scroll_filter = scope._kwargs.get("scroll_filter", {})

            scope._span.set_attribute(
                SemanticConvention.DB_COLLECTION_NAME, collection_name
            )
            scope._span.set_attribute(
                SemanticConvention.DB_QUERY_TEXT, str(scroll_filter)
            )
            scope._span.set_attribute(SemanticConvention.DB_FILTER, str(scroll_filter))

            scope._span.set_attribute(
                SemanticConvention.DB_QUERY_SUMMARY,
                f"{scope._db_operation} {collection_name} filter={scroll_filter}",
            )

        elif endpoint in ["qdrant.search", "qdrant.search_groups"]:
            query_vector = scope._kwargs.get("query_vector", [])
            limit = scope._kwargs.get("limit", 10)

            scope._span.set_attribute(
                SemanticConvention.DB_COLLECTION_NAME, collection_name
            )
            scope._span.set_attribute(
                SemanticConvention.DB_QUERY_TEXT, str(query_vector)
            )
            scope._span.set_attribute(SemanticConvention.DB_VECTOR_QUERY_TOP_K, limit)

            scope._span.set_attribute(
                SemanticConvention.DB_QUERY_SUMMARY,
                f"{scope._db_operation} {collection_name} limit={limit}",
            )

        elif endpoint == "qdrant.recommend":
            positive = scope._kwargs.get("positive", [])
            negative = scope._kwargs.get("negative", [])

            scope._span.set_attribute(
                SemanticConvention.DB_COLLECTION_NAME, collection_name
            )
            query_content = f"positive:{positive} negative:{negative}"
            scope._span.set_attribute(SemanticConvention.DB_QUERY_TEXT, query_content)

            scope._span.set_attribute(
                SemanticConvention.DB_QUERY_SUMMARY,
                f"{scope._db_operation} {collection_name} "
                f"positive={object_count(positive)} "
                f"negative={object_count(negative)}",
            )

        elif endpoint in ["qdrant.query_points", "qdrant.query_batch_points", "qdrant.query_points_groups"]:
            query = scope._kwargs.get("query", {})
            limit = scope._kwargs.get("limit", 10)
            using = scope._kwargs.get("using", None)
            prefetch = scope._kwargs.get("prefetch", [])
            query_filter = scope._kwargs.get("query_filter", None)

            scope._span.set_attribute(
                SemanticConvention.DB_COLLECTION_NAME, collection_name
            )

            query_text, prefetch_filters = format_query_points_ast(query, prefetch)
            scope._span.set_attribute(SemanticConvention.DB_QUERY_TEXT, query_text)
            scope._span.set_attribute(SemanticConvention.DB_VECTOR_QUERY_TOP_K, limit)

            if query_filter:
                scope._span.set_attribute(SemanticConvention.DB_FILTER, str(query_filter)[:500])
            elif prefetch_filters:
                scope._span.set_attribute(SemanticConvention.DB_FILTER, "; ".join(prefetch_filters)[:500])

            # Capture response output
            if scope._response:
                response_output = format_query_response(scope._response)
                scope._span.set_attribute(SemanticConvention.DB_RESPONSE_RETURNED_ROWS,
                    len(getattr(scope._response, "points", []) or []))
                scope._span.set_attribute("db.response.output", response_output)

        elif endpoint in ["qdrant.query", "qdrant.query_batch"]:
            query_text = scope._kwargs.get("query_text", "")
            limit = scope._kwargs.get("limit", 10)
            query_filter = scope._kwargs.get("query_filter", None)

            scope._span.set_attribute(
                SemanticConvention.DB_COLLECTION_NAME, collection_name
            )
            scope._span.set_attribute(SemanticConvention.DB_QUERY_TEXT, str(query_text)[:500])
            scope._span.set_attribute(SemanticConvention.DB_VECTOR_QUERY_TOP_K, limit)
            if query_filter:
                scope._span.set_attribute(SemanticConvention.DB_FILTER, str(query_filter)[:500])

            # Capture response output
            if scope._response:
                response_output = format_query_response(scope._response)
                scope._span.set_attribute(SemanticConvention.DB_RESPONSE_RETURNED_ROWS,
                    len(getattr(scope._response, "points", []) or []))
                scope._span.set_attribute("db.response.output", response_output)

    # Handle index operations
    elif scope._db_operation == SemanticConvention.DB_OPERATION_CREATE_INDEX:
        collection_name = scope._kwargs.get("collection_name", "unknown")
        field_name = scope._kwargs.get("field_name", "unknown")

        scope._span.set_attribute(
            SemanticConvention.DB_COLLECTION_NAME, collection_name
        )
        scope._span.set_attribute(
            SemanticConvention.DB_QUERY_TEXT, f"Field: {field_name}"
        )

        scope._span.set_attribute(
            SemanticConvention.DB_QUERY_SUMMARY,
            f"{scope._db_operation} {collection_name} field={field_name}",
        )

    scope._span.set_status(Status(StatusCode.OK))

    # Record metrics using helper
    if not disable_metrics:
        record_db_metrics(
            metrics,
            SemanticConvention.DB_SYSTEM_QDRANT,
            scope._server_address,
            scope._server_port,
            environment,
            application_name,
            scope._start_time,
            scope._end_time,
            scope._db_operation,
        )


def process_qdrant_response(
    response,
    db_operation,
    server_address,
    server_port,
    environment,
    application_name,
    metrics,
    start_time,
    span,
    capture_message_content,
    disable_metrics,
    version,
    instance=None,
    args=None,
    endpoint=None,
    **kwargs,
):
    """
    Process Qdrant response and generate telemetry.
    """
    # Create scope object
    scope = type("GenericScope", (), {})()
    scope._span = span
    scope._response = response
    scope._db_operation = db_operation
    scope._server_address = server_address
    scope._server_port = server_port
    scope._start_time = start_time
    scope._kwargs = kwargs

    # Process the response
    common_qdrant_logic(
        scope,
        environment,
        application_name,
        metrics,
        capture_message_content,
        disable_metrics,
        version,
        instance=instance,
        endpoint=endpoint,
    )

    return response
