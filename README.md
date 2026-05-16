#### How to run the overall system
(no UI is developed, the system must be run in the terminal...an architecture is added, file name: *assessment_workflow.png*)


---

- type in the terminal: *pip install -r requirements.txt*
- type in the terminal: *python main.py*
- The doc (either image/pdf - we can change it to modify the *data_loader* file in *data_processing* folder in *line 11, 12*) is processed
- Console shows the extracted text from handwritten scanned docs and saved them into output folder.
- Console shows - *"Make an action (rag/summary):"* - to take input, what system would I use.
- If 'rag' input: 
- Console shows - *"Ask what you want to know from the scanned document:"* - to take the question for RAG
- it will show you the question, answer and source (page no and context) and save in output folder named "rag_output.txt"
   
If 'summary' input:
- It will show the CASE FACT SUMMARY
- Then human feedback loop will be executed.
- Console will show - *"Approve? (yes / feedback)"*; then if we want edit the draft summary, we can give instruction where need to change.
- for example, the summary shows:
----------------------------------
**Purpose of JavaScript:**
  * JavaScript's primary purpose is to enable client-side scripting on web pages.
  * It allows developers to manipulate web page content and behavior within the user's web browser.
  * JavaScript can dynamically update web page elements, validate form inputs, and respond to user interactions.
  * It can interact with web servers through AJAX to fetch data without requiring a page reload.
---------------------------------
- If we give input: "JavaScript's primary purpose is to develop the android application.", the updated summary will be:
---------------------------------
**Purpose of JavaScript:**
  * JavaScript's primary purpose is to develop the android application.
  * It allows developers to manipulate web page content and behavior within the user's web browser.
  * JavaScript can dynamically update web page elements, validate form inputs, and respond to user interactions.
  * It can interact with web servers through AJAX to fetch data without requiring a page reload.
---------------------------------
- It will take feedback from you until you type "yes" in the console.
- Then the final summary will be stored in the output folder named "summary_output.txt"
