# mCP Server API Definition

This document defines the RESTful API for the mCP (Cognitive Control Protocol) Server. The API is designed to be simple, consistent, and easy to use. It follows the principles of the OpenAPI Specification.

## 1. Authentication

All API endpoints require authentication. Clients must include an API key in the `Authorization` header of their requests.

`Authorization: Bearer <your-api-key>`

## 2. API Endpoints

### 2.1. Tasks

#### `POST /tasks`

Creates a new cognitive task.

*   **Request Body:**

    ```json
    {
      "workflow_id": "summarization_workflow",
      "input_data": {
        "text": "The text to be summarized..."
      },
      "coordination_mode": "Sequential"
    }
    ```

    *   `workflow_id` (string, required): The ID of the workflow to execute.
    *   `input_data` (object, required): The input data for the task.
    *   `coordination_mode` (string, optional, default: "Sequential"): The coordination mode to use. Can be one of `Sequential`, `Parallel`, `Hierarchical`, `Adaptive`.

*   **Response:**

    *   `202 Accepted`: If the task was successfully created. The response body will contain the task ID.

        ```json
        {
          "task_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        }
        ```

    *   `400 Bad Request`: If the request body is invalid.
    *   `401 Unauthorized`: If the API key is missing or invalid.

#### `GET /tasks/{task_id}`

Retrieves the status of a cognitive task.

*   **Path Parameters:**

    *   `task_id` (string, required): The ID of the task.

*   **Response:**

    *   `200 OK`: The response body will contain the task status.

        ```json
        {
          "task_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
          "status": "running",
          "created_at": "2025-08-07T15:30:00Z"
        }
        ```

    *   `404 Not Found`: If the task does not exist.

#### `GET /tasks/{task_id}/result`

Retrieves the result of a completed cognitive task.

*   **Path Parameters:**

    *   `task_id` (string, required): The ID of the task.

*   **Response:**

    *   `200 OK`: The response body will contain the task result.

        ```json
        {
          "task_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
          "status": "completed",
          "result": {
            "summary": "This is the summary of the text."
          }
        }
        ```

    *   `202 Accepted`: If the task is still running.
    *   `404 Not Found`: If the task does not exist.
    *   `500 Internal Server Error`: If the task failed. The response body will contain an error message.

### 2.2. Protocols

#### `GET /protocols`

Retrieves a list of available cognitive protocols.

*   **Response:**

    *   `200 OK`: The response body will contain a list of protocols.

        ```json
        [
          {
            "name": "ReadabilityEnhancementProtocol",
            "version": "1.0.0",
            "type": "Aesthetic"
          },
          {
            "name": "SummarizerProtocol",
            "version": "1.2.0",
            "type": "Logical"
          }
        ]
        ```

This API provides the necessary functionality for clients to interact with the mCP Server in a simple and programmatic way.
