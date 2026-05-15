from data_processing.data_loader import extracted_text, extracted_text_list
from utils import save_as_text
from rag.rag import rag_pipeline
from summary_gen.summary2 import SummaryGenerator

def main():
    print("===================================")
    print("ORIGINAL TEXT EXTRACTED:")
    print(extracted_text)
    print("===================================")

    action = input("Make an action (rag/summary): ")

    if action == "rag":
        print(f"action : {action}")
        rag_pipeline(extracted_text_list)

    elif action == "summary":
        print(f"action : {action}")
        summary_generator = SummaryGenerator(extracted_text)
        final_output = summary_generator.generate()
        # print(final_output)

        save_as_text(final_output, "./output/summary_output.txt")

    else:
        print(f"{action} is an invalid action! please enter 'rag' or 'summary'...")



if __name__ == "__main__":
    main()