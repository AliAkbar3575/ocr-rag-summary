from summary_gen.summary1 import HITLSummaryEngine
from langgraph.types import Command


class SummaryGenerator:

    def __init__(self, text: str, thread_id: str = "summary-thread"):

        self.text = text
        self.engine = HITLSummaryEngine()

        self.config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

    def generate(self):

        result = self.engine.app.invoke(
            {"input_text": self.text},
            config=self.config
        )

        while "__interrupt__" in result:

            data = result["__interrupt__"][0].value

            print("\n====================")
            print("CURRENT SUMMARY:\n")
            print(data["case_fact_summary"])
            print("====================\n")

            user_input = input("Approve? (yes / feedback): ")

            result = self.engine.app.invoke(
                Command(resume=user_input),
                config=self.config
            )

        print("\nFINAL SUMMARY:\n")
        print(result["case_fact_summary"])

        return result["case_fact_summary"]