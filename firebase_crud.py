import firebase_admin
from firebase_admin import credentials, firestore, storage


def uploadFileReturnUrl(fileName):
    cred = credentials.Certificate('assets/solar-glare-firebase-adminsdk-qrutq-5d40af353c.json')
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'solar-glare.appspot.com'
    })

    # bucket = storage.bucket()
    bucket = storage.bucket()

    blob = bucket.blob('result/'+fileName)
    blob.upload_from_filename('assets/'+fileName)

    # Opt : if you want to make public access from the URL
    blob.make_public()

    print("your file url", blob.public_url)
    return blob.public_url
