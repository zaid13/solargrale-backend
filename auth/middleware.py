@app.middleware("http")

async def auth_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    print(f"check request url path is {request.url.path}")
    if request.url.path in ["/", "/docs", "/openapi.json", "/redoc", "/test_redis", "/send_campaign_verify_email",
                            "/verify_code_for_campaign", "/start_campaign_interview", "/next_campaign_question"]:
        return await call_next(request)

    campaign_status_regex = r'^/campaign_status(/[^/]*)?/?$'

    if re.match(campaign_status_regex, request.url.path):
        return await call_next(request)

    authorization_header = request.headers.get("Authorization")

    if authorization_header is None or not authorization_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Authorization header missing or invalid."
        )

    id_token = authorization_header.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(id_token)
        request.state.user = decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token: {}".format(str(e)))

    return await call_next(request)