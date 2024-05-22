from rest_framework.authentication import BaseAuthentication
from rest_framework import exceptions
import jwt
from firebase_admin import auth
from channels.middleware import BaseMiddleware
from channels.db import database_sync_to_async
from collections import namedtuple
# from loguru import logger as log


_TEST_MASTER_TOKEN = "master-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTIzNDU2Nzg5LCJuYW1lIjoiSm9zZXBoIn0" \
                     ".OpOSSw7e485LOP5PrzScxHb7SR6sAOMRckfFwi4rp7o"
_TEST_BUSINESS_TOKEN = "business-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTIzNDU2Nzg5LCJuYW1lIjoiSm9zZXBoIn0" \
                     ".OpOSSw7e485LOP5PrzScxHb7SR6sAOMRckfFwi4rp7o"
_TEST_FLEET_TOKEN = "fleet-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTIzNDU2Nzg5LCJuYW1lIjoiSm9zZXBoIn0" \
                     ".OpOSSw7e485LOP5PrzScxHb7SR6sAOMRckfFwi4rp7o"
_TEST_PASSENGER_TOKEN = "passenger-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTIzNDU2Nzg5LCJuYW1lIjoiSm9zZXBoIn0" \
                     ".OpOSSw7e485LOP5PrzScxHb7SR6sAOMRckfFwi4rp7o"
_TEST_DRIVER_TOKEN = "driver-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTIzNDU2Nzg5LCJuYW1lIjoiSm9zZXBoIn0" \
                     ".OpOSSw7e485LOP5PrzScxHb7SR6sAOMRckfFwi4rp7o"


User = namedtuple('User', 'user_id user_type')


class CustomFirebaseAuthentication(BaseAuthentication):

    def authenticate(self, request):

        if hasattr(request, 'path'):
            if request.path.startswith('/admin/'):
                print("path: {}".format(request.path))
                return None
            if request.path.startswith('/swagger/'):
                print("path: {}".format(request.path))
                return None
            if request.path.startswith('/redoc/'):
                print("path: {}".format(request.path))
                return None

        auth_header = request.META.get('HTTP_AUTHORIZATION', '')
        if not auth_header or 'Bearer ' not in auth_header:
            return None

        token = auth_header.split(' ')[1]

        if token == _TEST_MASTER_TOKEN:
            request.user_id = "test-master"
            request.user_type = "Master"
            return User(request.user_id, request.user_type), None
        if token == _TEST_BUSINESS_TOKEN:
            request.user_id = "test-business"
            request.user_type = "Business"
            return User(request.user_id, request.user_type), None
        if token == _TEST_FLEET_TOKEN:
            request.user_id = "test-fleet"
            request.user_type = "Fleet"
            return User(request.user_id, request.user_type), None
        if token == _TEST_PASSENGER_TOKEN:
            request.user_id = "test-passenger"
            request.user_type = "Passenger"
            return User(request.user_id, request.user_type), None
        if token == _TEST_DRIVER_TOKEN:
            request.user_id = "test-driver"
            request.user_type = "Driver"
            return User(request.user_id, request.user_type), None

        try:
            decoded_token = auth.verify_id_token(token)
            request.user_id = decoded_token['uid']
            user_type = decoded_token.get('user_type', None)  # Extract user role
            if user_type not in ('Fleet', 'Master', 'Passenger', 'Driver', 'Business'):
                request.user_type = "UnregisteredUser"
                # raise exceptions.AuthenticationFailed('Invalid user type')
            else:
                request.user_type = user_type
            return User(request.user_id, request.user_type), None
        except (jwt.exceptions.ExpiredSignatureError, ValueError, KeyError) as e:
            raise exceptions.AuthenticationFailed('Token is invalid {}'.format(e))
        except jwt.InvalidTokenError:
            raise exceptions.AuthenticationFailed('Invalid token')

