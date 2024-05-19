from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from google.auth.transport import requests
from google.oauth2 import id_token
from . import config

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

CLIENT_ID = config.CLIENT_ID


async def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        id_info = id_token.verify_oauth2_token(token, requests.Request(), CLIENT_ID)

        if id_info['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')

        return id_info

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
