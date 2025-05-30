import pytest
from unittest.mock import patch, MagicMock
from app.services.agent_service import ReActAgent, Thought, Observation
from app.services.tool_service import IntentClassifierTool, EmailTool, PharmacyAPITool
import json

@pytest.fixture
def mock_pharmacy_api():
    mock_api = MagicMock(spec=PharmacyAPITool)
    mock_api.get_pharmacy_by_phone.return_value = {
        "phone": "1-555-123-4567",
        "name": "HealthFirst Pharmacy",
        "city": "New York",
        "state": "NY",
        "prescriptions": [
            {"drug": "Lisinopril", "count": 42},
            {"drug": "Atorvastatin", "count": 38}
        ]
    }
    return mock_api

@pytest.fixture
def mock_openai_client():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    
    def mock_create(*args, **kwargs):
        messages = kwargs.get('messages', [])
        system_prompt = next((msg['content'] for msg in messages if msg['role'] == 'system'), '')
        
        if "think through what needs to be done next" in system_prompt.lower():
            mock_response.choices[0].message.content = json.dumps({
                "reasoning": "User is asking about services, need to classify intent first",
                "next_action": "classify user intent",
                "tool_name": "classify_intent",
                "tool_args": {
                    "user_message": "What services do you offer?",
                    "conversation_context": {"conversation_length": 1}
                }
            })
        elif "generate a natural response" in system_prompt.lower():
            mock_response.choices[0].message.content = (
                "That's a great question! Improving operations is crucial for any pharmacy. "
                "We offer several services that can significantly enhance efficiency and "
                "patient care. \n\n"
                "1. **Prescription Processing Automation**: This service can help you "
                "streamline your workflow, leading to a 30% reduction in processing time. By "
                "minimizing manual input, you'll also reduce errors, which enhances patient "
                "satisfaction.\n\n"
                "2. **Inventory Management**: Our inventory management tools allow you to "
                "keep track of stock levels in real-time, which can lead to a 15-20% cost "
                "savings. This way, you can prevent overstocking or running out of essential "
                "medications.\n\n"
                "3. **Compliance Tracking**: Staying compliant with regulations can be "
                "challenging. Our compliance tracking tools not only help you stay on top of "
                "necessary documentation and audits but can also ensure that you're following "
                "best practices consistently.\n\n"
                "4. **Patient Communication Tools**: Effective communication with patients is "
                "key to improving their experience. Our tools enhance patient engagement "
                "through reminders for refills, updates about potential drug interactions, "
                "and even health tips.\n\n"
                "I'd love to learn more about your current operations. What specific "
                "challenges are you facing that you're looking to improve?"
            )
        return mock_response
    
    mock_client.chat.completions.create.side_effect = mock_create
    return mock_client

@pytest.fixture
def mock_intent_classifier():
    mock_classifier = MagicMock(spec=IntentClassifierTool)
    mock_classifier.classify_intent.return_value = {"intent": "inquiry_services", "confidence": 0.95}
    return mock_classifier

@pytest.fixture
def mock_email_tool():
    mock_tool = MagicMock(spec=EmailTool)
    mock_tool.generate_email.return_value = {
        "to": "test@example.com",
        "subject": "Information About Our Pharmacy Services",
        "body": "Thank you for your interest in our services...",
        "topics": ["automation", "inventory"]
    }
    mock_tool.send_email.return_value = {"success": True, "message": "Email sent successfully"}
    return mock_tool

@pytest.fixture
def agent(mock_openai_client, mock_intent_classifier, mock_email_tool, mock_pharmacy_api):
    with patch('app.services.agent_service.OpenAI', return_value=mock_openai_client), \
         patch('app.services.agent_service.IntentClassifierTool', return_value=mock_intent_classifier), \
         patch('app.services.agent_service.EmailTool', return_value=mock_email_tool), \
         patch('app.services.agent_service.PharmacyAPITool', return_value=mock_pharmacy_api):
        agent = ReActAgent()
        return agent

def test_handle_service_inquiry(agent):
    """Test that the agent can handle inquiries about pharmacy services"""
    response = agent.process_message("What services do you offer for pharmacies?")

    assert "prescription processing automation" in response.lower()
    assert "inventory management" in response.lower()
    assert "compliance tracking" in response.lower()
    assert "patient communication tools" in response.lower()

    assert "30%" in response
    assert "15-20%" in response

def test_handle_service_inquiry_with_known_pharmacy(agent, mock_pharmacy_api):
    """Test that the agent can handle service inquiries from a known pharmacy"""
    mock_pharmacy_api.get_pharmacy_by_phone.return_value = {
        "phone": "1-555-123-4567",
        "name": "HealthFirst Pharmacy",
        "city": "New York",
        "state": "NY",
        "prescriptions": [
            {"drug": "Lisinopril", "count": 42},
            {"drug": "Atorvastatin", "count": 38}
        ]
    }

    agent.handle_incoming_call("1-555-123-4567")
    response = agent.process_message("What services can help us improve our operations?")

    assert "healthfirst pharmacy" in response.lower()
    assert "new york" in response.lower()
    assert "lisinopril" in response.lower()
    assert "atorvastatin" in response.lower()

def test_handle_service_inquiry_with_email_request(agent, mock_intent_classifier):
    """Test that the agent can handle service inquiries followed by email requests"""
    mock_intent_classifier.classify_intent.side_effect = [
        {"intent": "inquiry_services", "confidence": 0.95},
        {"intent": "request_email", "confidence": 0.95}
    ]

    response1 = agent.process_message("What services do you offer?")
    assert "prescription processing automation" in response1.lower()

    response2 = agent.process_message("Can you send me more information about these services?")
    assert "email" in response2.lower()
    assert "sent" in response2.lower()

def test_handle_service_inquiry_with_unknown_pharmacy(agent, mock_pharmacy_api, mock_intent_classifier):
    """Test that the agent can handle service inquiries from an unknown pharmacy"""
    mock_pharmacy_api.get_pharmacy_by_phone.return_value = None
    mock_intent_classifier.classify_intent.side_effect = [
        {"intent": "provide_info", "confidence": 0.95},
        {"intent": "provide_info", "confidence": 0.95},
        {"intent": "provide_info", "confidence": 0.95},
        {"intent": "provide_info", "confidence": 0.95}
    ]

    initial_response = agent.handle_incoming_call("1-555-999-8888")
    assert "name of your pharmacy" in initial_response.lower()

    response2 = agent.process_message("City Pharmacy")
    assert "city pharmacy" in response2.lower()
    assert "where is your pharmacy located" in response2.lower()
    
    response3 = agent.process_message("We're in Chicago, Illinois")
    assert "prescriptions" in response3.lower()
    assert "monthly" in response3.lower()
    
    response4 = agent.process_message("We process about 1000 prescriptions monthly")
    assert "1000" in response4
    assert "email" in response4.lower()

def test_react_loop(agent, mock_openai_client):
    """Test that the ReAct loop works correctly"""
    # Mock the thought generation to simulate a complete ReAct cycle
    mock_openai_client.chat.completions.create.side_effect = [
        # First thought: classify intent
        MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
            "reasoning": "Need to classify user intent",
            "next_action": "classify intent",
            "tool_name": "classify_intent",
            "tool_args": {
                "user_message": "test message",
                "conversation_context": {"conversation_length": 1}
            }
        })))]),
        # Second thought: generate email
        MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
            "reasoning": "User requested email",
            "next_action": "generate email",
            "tool_name": "generate_email",
            "tool_args": {
                "pharmacy_info": {"name": "Test Pharmacy"},
                "user_query": "test message",
                "topics_of_interest": ["automation"]
            }
        })))]),
        # Final response
        MagicMock(choices=[MagicMock(message=MagicMock(content="Email sent successfully"))])
    ]

    response = agent.process_message("Can you send me more information?")
    assert "email" in response.lower()
    assert "sent" in response.lower()

def test_tool_execution(agent):
    """Test that tools are executed correctly"""
    thought = Thought(
        reasoning="Test tool execution",
        next_action="calculate rx volume",
        tool_name="calculate_rx_volume",
        tool_args={"pharmacy": {"prescriptions": [{"count": 10}, {"count": 20}]}}
    )
    
    observation = agent._execute_tool(thought)
    assert observation.success
    assert observation.result == 30

def test_tool_execution_with_missing_context(agent):
    """Test that tool execution fails when required context is missing"""
    thought = Thought(
        reasoning="Test missing context",
        next_action="generate email",
        tool_name="generate_email",
        tool_args={"pharmacy_info": {"name": "Test"}}  # Missing required context
    )
    
    observation = agent._execute_tool(thought)
    assert not observation.success
    assert "Missing required context" in observation.error

def test_state_update(agent):
    """Test that agent state is updated correctly"""
    thought = Thought(
        reasoning="Test state update",
        next_action="find pharmacy",
        tool_name="find_pharmacy",
        tool_args={"phone_number": "1-555-123-4567"}
    )
    
    observation = Observation(
        result={"name": "Test Pharmacy", "city": "Test City"},
        success=True
    )
    
    agent._update_state(thought, observation)
    assert agent.current_pharmacy == observation.result
    assert len(agent.conversation_history) > 0
    assert "Thought: Test state update" in agent.conversation_history[-1]["content"] 