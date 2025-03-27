from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import time

from openai import OpenAI

from app.services.tool_service import IntentClassifierTool, EmailTool, PharmacyAPITool

client = OpenAI(api_key="api_key_here")

class AgentState(Enum):
    INITIAL = "initial"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"

@dataclass
class Tool:
    name: str
    description: str
    function: callable
    required_context: List[str]

@dataclass
class Thought:
    reasoning: str
    next_action: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None

@dataclass
class Observation:
    result: Any
    success: bool
    error: Optional[str] = None

class ReActAgent:
    def __init__(self):
        self.openai_client = client
        self.pharmacy_api = PharmacyAPITool()
        self.intent_classifier = IntentClassifierTool(self.openai_client)
        self.email_tool = EmailTool(self.openai_client)
        self.conversation_history = []
        self.current_pharmacy = None
        self.collected_info = {}
        self.topics_of_interest = []
        self.state = AgentState.INITIAL
        self.email_sent = False

        self.tools = [
            Tool(
                name="find_pharmacy",
                description="Look up a pharmacy by phone number",
                function=self.pharmacy_api.get_pharmacy_by_phone,
                required_context=["phone_number"]
            ),
            Tool(
                name="classify_intent",
                description="Determine the user's intent from their message",
                function=self.intent_classifier.classify_intent,
                required_context=["user_message", "conversation_context"]
            ),
            Tool(
                name="generate_email",
                description="Generate and send an email with pharmacy service information",
                function=self.email_tool.generate_email,
                required_context=["pharmacy_info", "user_query", "topics_of_interest"]
            ),
            Tool(
                name="send_email",
                description="Send the generated email",
                function=self.email_tool.send_email,
                required_context=["email_data"]
            ),
            Tool(
                name="calculate_rx_volume",
                description="Calculate total prescription volume for a pharmacy",
                function=self.calculate_total_rx_volume,
                required_context=["pharmacy"]
            )
        ]

    def calculate_total_rx_volume(self, pharmacy: Optional[Dict[str, Any]]) -> int:
        """Calculate total prescription volume for a pharmacy"""
        if not pharmacy or "prescriptions" not in pharmacy:
            return 0
        return sum(rx.get("count", 0) for rx in pharmacy["prescriptions"])

    def _generate_thought(self, observation: Optional[Observation] = None) -> Thought:
        """Generate the next thought based on current state and observation"""
        system_prompt = """
        You are a ReAct agent for a pharmacy services sales team. Your goal is to give information to try and sell them
        your pharmacy AI sales platform.
        
        Current state:
        - Conversation history: {conversation_history}
        - Current pharmacy: {current_pharmacy}
        - Collected info: {collected_info}
        - Topics of interest: {topics_of_interest}
        - Email already sent: {email_sent}
        
        Available tools:
        {tools}
        
        Previous observation:
        {observation}
        
        Think through what needs to be done next and choose the most appropriate tool.
        If you've already sent an email, DO NOT send another one.
        
        Format your response as JSON with the following structure:
        {{
            "reasoning": "your reasoning here",
            "next_action": "what you plan to do next",
            "tool_name": "name of the tool to use (if any)",
            "tool_args": {{arguments for the tool if needed}}
        }}
        
        IMPORTANT: If the user has requested information by email and you've already sent it or 
        are about to send it, set next_action to "continue_conversation".
        """
        tools_desc = "\n".join([
            f"- {tool.name}: {tool.description} (requires: {', '.join(tool.required_context)})"
            for tool in self.tools
        ])

        obs_desc = "None"
        if observation:
            obs_desc = f"Result: {observation.result}\nSuccess: {observation.success}"
            if observation.error:
                obs_desc += f"\nError: {observation.error}"

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt.format(
                        conversation_history=self.conversation_history,
                        current_pharmacy=self.current_pharmacy,
                        collected_info=self.collected_info,
                        topics_of_interest=self.topics_of_interest,
                        tools=tools_desc,
                        observation=obs_desc,
                        email_sent=self.email_sent
                    )}
                ],
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            try:
                print("Received response:", content)
                thought_data = json.loads(content)

                if "reasoning" not in thought_data or "next_action" not in thought_data:
                    raise ValueError("Missing required fields in thought response")

                if self.email_sent and thought_data.get("tool_name") == "send_email":
                    print("Email already sent, forcing continue_conversation")
                    thought_data["next_action"] = "continue_conversation"
                    thought_data["tool_name"] = None
                    thought_data["tool_args"] = {}

                return Thought(
                    reasoning=thought_data.get("reasoning", ""),
                    next_action=thought_data.get("next_action", "continue_conversation"),
                    tool_name=thought_data.get("tool_name"),
                    tool_args=thought_data.get("tool_args", {})
                )
            except json.JSONDecodeError as e:
                print(f"Error parsing thought JSON: {e}")
                print(f"Raw content: {content}")
                return Thought(
                    reasoning="Error parsing JSON response",
                    next_action="continue_conversation",
                    tool_name=None,
                    tool_args={}
                )
            except ValueError as e:
                print(f"Error validating thought data: {e}")
                return Thought(
                    reasoning="Invalid thought data structure",
                    next_action="continue_conversation",
                    tool_name=None,
                    tool_args={}
                )

        except Exception as e:
            print(f"Error generating thought: {e}")
            return Thought(
                reasoning="Error in thought generation",
                next_action="continue_conversation",
                tool_name=None,
                tool_args={}
            )

    def _execute_tool(self, thought: Thought) -> Observation:
        """Execute the chosen tool with the provided arguments"""
        if not thought.tool_name:
            return Observation(result=None, success=True)

        tool = next((t for t in self.tools if t.name == thought.tool_name), None)
        if not tool:
            return Observation(
                result=None,
                success=False,
                error=f"Tool {thought.tool_name} not found"
            )

        tool_args = thought.tool_args or {}

        missing_context = [
            arg for arg in tool.required_context
            if arg not in tool_args
        ]
        if missing_context:
            return Observation(
                result=None,
                success=False,
                error=f"Missing required context: {', '.join(missing_context)}"
            )

        try:
            result = tool.function(**tool_args)

            if thought.tool_name == "send_email" and result.get("success", False):
                self.email_sent = True
                print("Email status updated: email_sent =", self.email_sent)

            return Observation(result=result, success=True)
        except Exception as e:
            return Observation(
                result=None,
                success=False,
                error=str(e)
            )

    def _update_state(self, thought: Thought, observation: Observation):
        """Update the agent's state based on the thought and observation"""
        if thought.tool_name:
            self.conversation_history.append({
                "role": "assistant",
                "content": f"Thought: {thought.reasoning}\nAction: {thought.next_action}\nResult: {observation.result if observation.success else observation.error}"
            })

        if thought.tool_name == "find_pharmacy" and observation.success:
            self.current_pharmacy = observation.result
        elif thought.tool_name == "classify_intent" and observation.success:
            intent_data = observation.result
            if intent_data.get("intent") == "provide_info":
                self.collected_info.update(intent_data.get("info", {}))
            elif intent_data.get("intent") == "request_email":
                self.collected_info["requested_email"] = True

        if thought.tool_name == "generate_email" and observation.success:
            email_data = observation.result
            if "topics" in email_data:
                self.topics_of_interest.extend(email_data["topics"])

    def _validate_tool_inputs(self, tool: Tool, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate tool inputs before execution"""
        missing_context = [
            arg for arg in tool.required_context
            if arg not in args
        ]
        if missing_context:
            return False, f"Missing required context: {', '.join(missing_context)}"

        return True, None

    def _generate_natural_response(self, thought: Thought, observation: Observation) -> str:
        """Generate a natural response based on the thought and observation"""
        system_prompt = """
        You are a pharmacy services sales agent. Generate a natural, helpful response based on:
        
        Thought: {thought}
        Observation: {observation}
        
        Current state:
        - Conversation history: {conversation_history}
        - Current pharmacy: {current_pharmacy}
        - Collected info: {collected_info}
        - Topics of interest: {topics_of_interest}
        - Email already sent: {email_sent}
        
        The response should be:
        1. Natural and conversational
        2. Address the user's needs
        3. Include relevant information from the observation
        4. Guide the conversation forward
        
        If you have sent an email, make sure to tell the user that you've already sent the information to their email.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt.format(
                        thought=thought.reasoning,
                        observation=f"Result: {observation.result}\nSuccess: {observation.success}",
                        conversation_history=self.conversation_history,
                        current_pharmacy=self.current_pharmacy,
                        collected_info=self.collected_info,
                        topics_of_interest=self.topics_of_interest,
                        email_sent=self.email_sent
                    )}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating natural response: {e}")
            return "I apologize, but I'm having trouble generating a response. Could you please rephrase your question?"

    def process_message(self, message: str) -> str:
        """Process a user message using the ReAct loop"""
        try:
            self.conversation_history.append({"role": "user", "content": message})
            thought = self._generate_thought()
            observation = None

            max_iterations = 5
            iteration_count = 0

            while thought.next_action != "continue_conversation" and iteration_count < max_iterations:
                iteration_count += 1
                print(f"Loop iteration {iteration_count}/{max_iterations}")

                if thought.tool_name:
                    tool = next((t for t in self.tools if t.name == thought.tool_name), None)
                    if not tool:
                        observation = Observation(
                            result=None,
                            success=False,
                            error=f"Tool {thought.tool_name} not found"
                        )
                    else:
                        is_valid, error = self._validate_tool_inputs(tool, thought.tool_args or {})
                        if not is_valid:
                            observation = Observation(result=None, success=False, error=error)
                        else:
                            observation = self._execute_tool(thought)

                self._update_state(thought, observation)
                thought = self._generate_thought(observation)

            if iteration_count >= max_iterations:
                print(f"WARNING: Reached maximum iterations ({max_iterations}), forcing exit from loop")

            if observation is None:
                observation = Observation(result=None, success=True)

            response = self._generate_natural_response(thought, observation)
            self.conversation_history.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            print(f"Error in ReAct loop: {e}")
            return "I apologize, but I encountered an error processing your message. Please try again."

    def handle_incoming_call(self, caller_phone: str) -> str:
        """Initial handling of an incoming call based on phone number"""
        thought = Thought(
            reasoning="Need to identify the calling pharmacy",
            next_action="find_pharmacy",
            tool_name="find_pharmacy",
            tool_args={"phone_number": caller_phone}
        )

        observation = self._execute_tool(thought)
        self._update_state(thought, observation)

        if observation.success and observation.result:
            pharmacy = observation.result
            rx_volume = self.calculate_total_rx_volume(pharmacy)
            greeting = f"Hello, thanks for calling! I see you're calling from {pharmacy['name']}. "
            greeting += f"I notice you handle about {rx_volume} prescriptions. How can I assist you today with our pharmacy services?"
        else:
            greeting = "Hello, thanks for calling our pharmacy services! I'd be happy to tell you about how we can help your pharmacy. Could I get the name of your pharmacy, please?"

        self.conversation_history.append({"role": "assistant", "content": greeting})
        return greeting

def simulate_conversation():
    """Simulate a conversation with the chatbot"""
    agent = ReActAgent()
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