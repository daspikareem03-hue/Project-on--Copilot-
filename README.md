# Project-on--Copilot-
Instamart Customer intelligence and support Copilot 
# project overview 

      The swiggy instamart customer intelligence and support copilot is an AI and dataset driven project is used to improve the customer experience and support system



# this project is divided into four parts 

part A - Data cleaning and data cretion 
part B - customer segmentation and chrun predition 
part C - support ticket classification and routing 
part D - policy RAG assistant and refund agent 
s
# requirement libery 
pandas
numpy
scikit-learn
joblib
SQL lite 

# project on copilot struture

project on copilot

    data
       raw
       clean
       policies
       instamart_db 
    part_a.py
    outputs 
        customer_segments.csv
        kmeans_evalution.csv
        lapse_feature.csv
        laspe_model.joblib
        retention_list.csv
        rfm.csv
        ticket_routing.csv
        triage_results.csv
        triage_budget_demo.csv
        triage_retry_log.txt
    state
        987fe2f0-1e13-4c02-96a0-1c3532237cf7.json
    part_b.py
    part_c.py
    part_c_triage.py
    part_d.py
    generate_data.py
    mock_llm.py
    README.md 
    requirement.txt


 # part A - Data cleaning and data cretion 
         This part is focused on cleaning the datasets from the raw data creating the instamart.db for making dataset in orangazined mannner, the data is cleaned by handling duplicate records , missing values, inconsistent spellings, and invalid values. 
         The cleaned data is stored in SQL lite for analaysis purpose

# part B - customer segmentation and chrun predition 
        
                This part is used groups or seperate the customers database and make it clear view for analaysis and also praparing the chrum prediction for risk analaysis .

                customer chrun risk summary: chrun risk 
                     high - 253 , medium - 215, low- 732

                customer features are created. 


# part C - support ticket classification and routing 
            This part processes customer support tickets and classifies them into relevant categories and the system supports privacy protection, ticket routing and retry handling and data related to ticket are categoried
            The results are stored in outputs file. 

            Ticket category counts : true_category delivery- 43 , fraud- 39, billing-39, technical-39
            after this the ticket cleaning is over remaining tickets are 160 and also categories the tickets rounting summary 


# part D - policy RAG assistant and refund agent 
           This part encompasess two major components 
               policy RAG assistant
               React refund support agent 

    In policy RAG assistant  
            The policy assistant rettrivals relevant information from five policy documents : 
                         delivery_policy.txt
                         escalation_poilcy.txt
                         payment_policy.txt
                         privacy_policy.txt
                         refund_policy.txt
            This documents divided into chunks and relevant poilcy chunks and generates an answer based on retrieved information.
    In refund support agent 
             the refund agent uses a react - style workflow to inspect orders , retrieve policies, calculate refunds and decide the refund can be isssued or escalated 
            
            the agent follows some rules foe satify purpose :
                      Only completed orders may be refunded
                      late delivery is defined as delivery taking more than 45 mintes 
                      refunds up to INR 1000 can be apporved automatically
                      refunds above INR 1000 require human approval 

           Here we are testing the refund tools orders_id : ORD-110093 and its work correctly. 


# privacy and safety 

       The project uses the mock llm for testing  
       No real network call are used like API Key
       overall this project is run very safe and secure and considered during support ticket processing and agent 


# conclusion 

          This swiggy instamart customer intelligence and support copilot combies the dta analytics , machine learning , privacy - aware ticket processing , policy retrival, and refund decision support agent The project aims to improve customer support efficiency while maintaining policy grounding, privacy, and safe refund handling.