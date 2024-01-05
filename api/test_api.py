#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:10:05 2023

@author: smgai
"""

import json 
import requests

def call_API():

    URL = "http://192.168.137.86:5110/items"
    API_KEY = "failure_"

    # Header
    headers  = {'Content-Type': 'application/json'}
                # 'Authorization': f'Bearer {API_KEY}'}

    # Body
    params = {
        "name": "foo",
        "description": "111222",
        "price": 0,
        "tax": 0,
        "tags": []
    }

    # Get the answer from the OpenAI
    response = requests.post(
        URL,
        headers = headers,
        data = json.dumps(params))
    
    if response.status_code != 200:
        print('error: ' + str(response.status_code))
    else:
        return response.json()


if __name__ == '__main__':
    
    # Input
    inp = """who are you """
    
    # way1. call from API
    res = call_API()
    
    # way2. call from Langchain
    # llm = LangchainCustomLLM()
    # res = llm(inp)

    print(f"\n> Question: {inp}")
    print(f"\n> Answer: {res}")

    A=1
