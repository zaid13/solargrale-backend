import firebase_admin
from firebase_admin import credentials, firestore, storage

cred = credentials.Certificate('assets/solar-glare-firebase-adminsdk-qrutq-5d40af353c.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'solar-glare.appspot.com'
})

def uploadFileReturnUrl(userId, simId, fileName,path ):


    bucket = storage.bucket()

    blob = bucket.blob('result/'+userId+'/'+simId+'/'+fileName)
    blob.upload_from_filename(path)

    # Opt : if you want to make public access from the URL
    blob.make_public()

    print("your file url", blob.public_url)
    return blob.public_url
