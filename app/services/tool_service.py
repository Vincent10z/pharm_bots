import json
from datetime import datetime
from typing import Dict, Any, List


class IntentClassifierTool:
    """Tool to determine the user's intent based on their message"""
    def __init__(self, openai_client):
        self.client = openai_client

    INTENTS = [
        "greeting",
        "inquiry_services",
        "inquiry_pricing",
        "inquiry_integration",
        "request_email",
        "request_callback",
        "provide_info",
        "farewell",
        "other"
    ]

    def classify_intent(self, user_message: str, conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the intent of the user message using OpenAI"""
        system_prompt = """
        You are an intent classifier for a pharmacy sales agent. Classify the user's intent into one of these categories:
        - greeting: Initial greetings or hellos
        - inquiry_services: Questions about services offered
        - inquiry_pricing: Questions about pricing or costs
        - inquiry_integration: Questions about system integration
        - request_email: Requesting information via email
        - request_callback: Requesting a callback or phone meeting
        - provide_info: Providing information about their pharmacy
        - farewell: Ending the conversation
        - other: None of the above
        
        Return only the intent category and confidence (0-1) in JSON format like:
        {"intent": "category_name", "confidence": 0.95}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"Error in intent classification: {e}")
            return {"intent": "other", "confidence": 0.0}

class EmailTool:
    """Tool to generate and 'send' emails to pharmacy customers"""

    def __init__(self, openai_client):
        self.client = openai_client

    def generate_email(self, pharmacy_info: Dict[str, Any], user_query: str,
                       topics_of_interest: List[str]) -> Dict[str, str]:
        """Generate an email based on pharmacy information and their interests"""

        service_offerings = {
            "inventory": "Our inventory management system helps reduce waste and ensures you never run out of critical medications.",
            "automation": "Our prescription processing automation reduces manual entry errors by up to 80% and saves staff time.",
            "compliance": "Our compliance tracking system keeps you updated with changing regulations and helps prevent costly violations.",
            "analytics": "Our analytics dashboard gives you insights into prescription trends, patient demographics, and business performance."
        }

        relevant_offerings = []
        for topic in topics_of_interest:
            for key, description in service_offerings.items():
                if topic.lower() in key or key in topic.lower():
                    relevant_offerings.append(description)

        if not relevant_offerings:
            relevant_offerings = list(service_offerings.values())

        rx_volume = 0
        if pharmacy_info and "prescriptions" in pharmacy_info:
            for rx in pharmacy_info["prescriptions"]:
                rx_volume += rx.get("count", 0)

        if pharmacy_info and pharmacy_info.get("name"):
            pharmacy_name = pharmacy_info["name"]
            greeting = f"Dear {pharmacy_name} team,"
        else:
            greeting = "Dear Pharmacy Team,"

        pharmacy_location = ""
        if pharmacy_info:
            if pharmacy_info.get("city") and pharmacy_info.get("state"):
                pharmacy_location = f"as a pharmacy in {pharmacy_info['city']}, {pharmacy_info['state']}"

        try:
            system_prompt = f"""
            You are an email writer for a pharmacy services company. 
            Create a professional, concise follow-up email for a pharmacy that contacted our sales team.
            
            Use these details to personalize the email:
            - Pharmacy name: {pharmacy_info.get('name', 'your pharmacy')}
            - Location: {pharmacy_location}
            - Prescription volume: {rx_volume if rx_volume > 0 else 'your pharmacy'}
            
            Include these service offerings:
            {' '.join(relevant_offerings)}
            
            The email should:
            1. Be professional but warm
            2. Address their specific query: "{user_query}"
            3. Highlight how our services can help them based on their prescription volume
            4. Include a clear call to action
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate a follow-up email."}
                ]
            )

            email_body = response.choices[0].message.content

            email = {
                "to": pharmacy_info.get("email", "customer@example.com"),
                "subject": "Information About Our Pharmacy Services",
                "body": email_body
            }

            print(f"[MOCK] Email generated and ready to send to {email['to']}")

            return email

        except Exception as e:
            print(f"Error generating email: {e}")
            return {
                "to": pharmacy_info.get("email", "customer@example.com"),
                "subject": "Information About Our Pharmacy Services",
                "body": f"{greeting}\n\nThank you for your interest in our pharmacy services. We specialize in helping pharmacies like yours improve efficiency and reduce costs.\n\nI'd be happy to schedule a call to discuss how we can help your pharmacy specifically.\n\nBest regards,\nThe Sales Team"
            }

    def send_email(self, email_data: Dict[str, str]) -> Dict[str, Any]:
        """Mock sending an email"""
        print(f"[MOCK] Email sent to: {email_data['to']}")
        print(f"[MOCK] Subject: {email_data['subject']}")
        print(f"[MOCK] Body preview: {email_data['body'][:100]}...")

        return {
            "success": True,
            "message": f"Email sent successfully to {email_data['to']}",
            "timestamp": datetime.now().isoformat()
        }