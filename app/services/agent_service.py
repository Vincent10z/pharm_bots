from typing import Dict, List, Optional, Any, Tuple

from openai import OpenAI

from app.services.tool_service import IntentClassifierTool, EmailTool
from app.services.integration import PharmacyAPI

client = OpenAI(api_key="api_key_here")


class PharmacyAgent:
    def __init__(self):
        self.openai_client = client
        self.pharmacy_api = PharmacyAPI()
        self.intent_classifier = IntentClassifierTool(self.openai_client)
        self.email_tool = EmailTool(self.openai_client)
        self.conversation_history = []
        self.current_pharmacy = None
        self.collected_info = {}
        self.topics_of_interest = []

    def get_pharmacy_medication_orders(self, pharmacy_id: int) -> List[Dict[str, Any]]:
        """Get medication orders for a pharmacy"""
        pharmacy = self.current_pharmacy
        if not pharmacy:
            return []

        return pharmacy.get("prescriptions", [])

    def find_pharmacy_by_phone(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Look up a pharmacy by phone number using the API"""
        return self.pharmacy_api.get_pharmacy_by_phone(phone_number)

    def calculate_total_rx_volume(self, pharmacy: Optional[Dict[str, Any]]) -> int:
        """Calculate total prescription volume for a pharmacy"""
        if not pharmacy or "prescriptions" not in pharmacy:
            return 0
        return sum(rx.get("count", 0) for rx in pharmacy["prescriptions"])

    def generate_response(self, user_message: str) -> str:
        """Generate a response using the OpenAI model"""
        self.conversation_history.append({"role": "user", "content": user_message})
        system_prompt = ("You are a helpful pharmacy services sales agent representing Pharmesol, you answer any questions"
                         "about the service you offer to incoming calls from prospective pharmacies that want to inquire"
                         "about your pharmacy services you offer")

        if self.current_pharmacy:
            pharmacy_info = f"""
            You're speaking with {self.current_pharmacy['name']} from {self.current_pharmacy['city']}, {self.current_pharmacy['state']}.
            Their prescription volume is approximately {self.calculate_total_rx_volume(self.current_pharmacy)} prescriptions.
            Their most prescribed medications are: {', '.join([rx['drug'] for rx in self.current_pharmacy['prescriptions']])}.
            """
            system_prompt += pharmacy_info + ("always mention the pharmacy by name in your response and their lcoation, use"
                                              "all available information to make the caller feel noticed and welcome")

        system_prompt += """
        Be professional but conversational. Your goal is to understand their needs and guide them to our pharmacy services.
        
        Key services to emphasize:
        1. Prescription processing automation
        2. Inventory management
        3. Compliance tracking
        4. Patient communication tools
        
        Benefits to mention:
        - 30% reduction in processing time
        - 15-20% cost savings
        - Improved accuracy and patient satisfaction
        
        Always look for opportunities to gather information about their pharmacy and tailor your responses.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.conversation_history[-5:]
                ]
            )

            agent_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": agent_response})

            return agent_response

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize for the technical difficulties. How can I help you with your pharmacy needs today?"

    def determine_user_intent(self, user_message: str) -> Dict[str, Any]:
        """Use the intent classifier tool to determine user intent"""
        context = {
            "pharmacy": self.current_pharmacy,
            "collected_info": self.collected_info,
            "conversation_length": len(self.conversation_history)
        }
        return self.intent_classifier.classify_intent(user_message, context)

    def handle_email_request(self, user_message: str) -> Tuple[bool, str]:
        """Handle a request for email information using the email tool"""
        if not self.topics_of_interest:
            topics = ["inventory", "automation", "compliance", "analytics"]
            for message in self.conversation_history:
                if message["role"] == "user":
                    content = message["content"].lower()
                    for topic in topics:
                        if topic in content and topic not in self.topics_of_interest:
                            self.topics_of_interest.append(topic)

        if not self.topics_of_interest:
            self.topics_of_interest = ["general information", "services"]

        email_data = self.email_tool.generate_email(
            self.current_pharmacy,
            user_message,
            self.topics_of_interest
        )

        result = self.email_tool.send_email(email_data)

        if result["success"]:
            return True, f"I've sent you an email with detailed information about our pharmacy services to {email_data['to']}. Is there anything specific you'd like me to address in a follow-up call?"
        else:
            return False, "I'm sorry, I couldn't send the email at this time. Would you like to schedule a call with one of our specialists instead?"

    def handle_incoming_call(self, caller_phone: str) -> str:
        """Initial handling of an incoming call based on phone number"""
        self.current_pharmacy = self.find_pharmacy_by_phone(caller_phone)

        if self.current_pharmacy:
            pharmacy_name = self.current_pharmacy["name"]
            greeting = f"Hello, thanks for calling! I see you're calling from {pharmacy_name}. "
            rx_volume = self.calculate_total_rx_volume(self.current_pharmacy)
            greeting += f"I notice you handle about {rx_volume} prescriptions. How can I assist you today with our pharmacy services?"

            self.conversation_history.append({"role": "assistant", "content": greeting})
            return greeting
        else:
            greeting = "Hello, thanks for calling our pharmacy services! I'd be happy to tell you about how we can help your pharmacy. Could I get the name of your pharmacy, please?"

            self.conversation_history.append({"role": "assistant", "content": greeting})
            return greeting

    def process_message(self, user_message: str) -> str:
        """Process a user message, determine intent, and respond appropriately"""
        intent_data = self.determine_user_intent(user_message)
        intent = intent_data["intent"]

        if intent == "request_email":
            success, response = self.handle_email_request(user_message)
            return response

        if intent == "provide_info" and not self.current_pharmacy:
            if "name" not in self.collected_info and len(user_message) < 50:
                self.collected_info["name"] = user_message
                return f"Thanks for sharing that, {user_message}! Where is your pharmacy located (city and state)?"
            elif "location" not in self.collected_info and "name" in self.collected_info:
                self.collected_info["location"] = user_message
                return "Great! Approximately how many prescriptions does your pharmacy process monthly?"
            elif "rx_volume" not in self.collected_info and "location" in self.collected_info:
                try:
                    words = user_message.replace(",", "").split()
                    for word in words:
                        if word.isdigit():
                            self.collected_info["rx_volume"] = int(word)
                            break

                    if "rx_volume" in self.collected_info:
                        return f"Thank you! With {self.collected_info['rx_volume']} prescriptions, we can definitely help streamline your operations. Our services can help with inventory management, prescription processing automation, and compliance tracking. Would you like me to email you more information?"
                except:
                    pass

        return self.generate_response(user_message)


def simulate_conversation():
    """Simulate a conversation with the chatbot"""
    agent = PharmacyAgent()
    caller_phone = "1-555-123-4567"  # Known pharmacy
    # caller_phone = "1-555-999-8888"  # Unknown pharmacy

    print("Agent:", agent.handle_incoming_call(caller_phone))

    while True:
        user_input = input("Customer: ")
        if user_input.lower() in ["exit", "quit", "bye", "end", "see you", "goodbye"]:
            break

        response = agent.process_message(user_input)
        print("Agent:", response)

        if "goodbye" in response.lower() and "thank you" in response.lower():
            break

if __name__ == "__main__":
    simulate_conversation()