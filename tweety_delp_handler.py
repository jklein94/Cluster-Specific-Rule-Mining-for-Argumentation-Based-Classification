import requests
def call_tweety_delp(knowledge_base: str,query: str):
    payload['kb'] = knowledge_base
    payload['query'] = query

    response = requests.post(url, json=payload)

        # Check the response status code
    if response.status_code == 200:
        print(f"Request successful. Response: {response.json()['answer']}")
        return response.json(), response.status_code  # Print the response JSON
    else:
        print(f"Request failed with status code {response.status_code}:")
        return {'answer': None}, response.status_code  # Print the error response text

# Define the API endpoint URL
url = "http://localhost:8080/ping"  # Replace with your API endpoint URL

kb = 'Bird(opus). \
Penguin(tweety).\
Wings(tweety).\
Bird(X) <- Penguin(X).\
Fly(X) -< Bird(X).\
~Fly(X) -< Penguin(X).'


kb_icmes = \
"""
A
B
A && !C
D && C
!B && C
!A || C
B && !C
(D || B) && !A
!D && (F || !C)"""

# #with open('/home/jklein/dev/argumentation_based_classification/test_rules_zoo/delp/c_0_filtered_clusters_train_zoo_clean_rules.pl', 'r') as f:

#     example_kb = f.read()

# Define the JSON payload
payload = {
    "cmd": "query",
    "email": "jklein94@uni-koblenz.de",       # Replace with your email
    "compcriterion": "genspec",                  # Replace with your compcriterion ID
    "kb": kb,             # Replace with your knowledge base
    "query": 'Fly(tweety)' ,
    "timeout": 600,
    "unit_timeout": 'sec' 
                     
}
# payload = {
  
#     "cmd": "get_model",
#     "email": "pyarg@mail.com",
#     "nr_of_arguments": 20,
#     "attacks": [[1, 2], [2, 1], [3, 3], [3, 1]],
#     "semantics": "ad",
#     "solver": "simple",
#     "timeout": 10,
#     "unit_timeout": "ms"

# }


payload_icmes = {
   "cmd": "value",
   "email": "jklein94",
   "measure": "drastic",
   "kb": kb_icmes,
   "format": "tweety",
   "timeout": 100,
    "unit_timeout": "ms"
}



# def call_tweety_delp(knowledge_base: str,query: str):
#     payload['kb'] = knowledge_base
#     payload['query'] = query

#     response = requests.post(url, json=payload)

#         # Check the response status code
#     if response.status_code == 200:
#         print(f"Request successful. Response: {response.json()}")
#         return response.json(), response.status_code  # Print the response JSON
#     else:
#         print(f"Request failed with status code {response.status_code}:")
#         return response.text, response.status_code  # Print the error response text


ping = {
    "id": 1,
    "content": "Test ping" 
} 

if __name__ == '__main__':
    # Send the POST request with JSON payload
    response = requests.post(url, json=payload)

    # Check the response status code
    if response.status_code == 200:
        print("Request successful. Response:")
        print(response.json())  # Print the response JSON
    else:
        print(f"Request failed with status code {response.status_code}:")
        print(response.text)  # Print the error response text
