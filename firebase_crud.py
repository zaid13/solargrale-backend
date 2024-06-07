import firebase_admin
from firebase_admin import credentials, firestore, storage

cred = credentials.Certificate('assets/solar-glare-firebase-adminsdk-qrutq-5d40af353c.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'solar-glare.appspot.com'
})

def uploadFileReturnUrl(payload, fileName,path,uploadPath ):


    bucket = storage.bucket()

    blob = bucket.blob('result/'+payload.meta_data.user_id+'/'+payload.meta_data.sim_id+'/'+fileName)
    blob.upload_from_filename(path)

    # Opt : if you want to make public access from the URL
    blob.make_public()

    print("your file url", blob.public_url)
    uploadPath.put( blob.public_url)
    return blob.public_url
