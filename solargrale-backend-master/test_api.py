#
#
# import requests
# import json
#
# # Ersetzen Sie dies durch die tatsächliche URL Ihrer API
# api_url = "http://d2x.eu/api/process_glare"
#
# # Ersetzen Sie dies durch Ihre tatsächlichen Eingabedaten
# data = {
#     "identifier": "unique_identifier",  # Hier den Identifier einfügen
#     "pv_areas": [
#         [
#             {"latitude": 53.74621563, "longitude": 9.66355692, "ground_elevation": 0, "height_above_ground": 15},
#             {"latitude": 53.74610293, "longitude": 9.66353955, "ground_elevation": 0, "height_above_ground": 15},
#             {"latitude": 53.74609897, "longitude": 9.66358732, "ground_elevation": 0, "height_above_ground": 10},
#             {"latitude": 53.74620921, "longitude": 9.66360361, "ground_elevation": 0, "height_above_ground": 10}
#         ]
#     ],
#     "list_of_pv_area_information": [
#         {"azimuth": 90, "tilt": 20, "name": "Vertical PV area 1"}
#     ],
#     "list_of_ops": [
#         {"latitude": 53.74614449, "longitude": 9.66371593, "ground_elevation": 0, "height_above_ground": 10}
#     ],
#     "utc": 1
# }
#
# # Optionale Header, falls benötigt (z.B. für Authentifizierung)
# headers = {
#     "Content-Type": "application/json",
#     "API-Key": "toto901toto"
# }
#
# # Senden der POST-Anfrage
# response = requests.post(api_url, data=json.dumps(data), headers=headers)
#
# # Ausgabe der Antwort
# print("Status Code:", response.status_code)
# print("Response Body:", type(response.json()))
# res = response.json()
# print(res.keys())
#
# # pv_areas = res['pv_areas']
# daily_summary_dfs = res['daily_summary_dfs']['date']
# identifier = res['identifier']
# simplified_dfs = res['simplified_dfs']
# print(res['identifier'])
# print(res['identifier'])
# print(res['identifier'])
#
