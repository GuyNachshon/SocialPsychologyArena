"""Experiment Orchestrator using LangGraph for multi-agent social psychology experiments."""

import asyncio
import json
import logging
import random
import time
import os
from typing import Dict, List, Any, Optional, Callable, TypedDict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv

from .dsl_parser import Scenario, Role, Metric, MetricType
from .metrics.metric_engine import MetricEngine

# Import image generation utilities
try:
    from .utils.image_generator import generate_asch_lines, generate_multiple_trials
    IMAGE_GENERATION_AVAILABLE = True
except ImportError:
    try:
        from .utils.simple_image_generator import generate_simple_asch_lines as generate_asch_lines, generate_multiple_simple_trials as generate_multiple_trials
        IMAGE_GENERATION_AVAILABLE = True
        logger.info("Using simple text-based image generator (matplotlib not available)")
    except ImportError:
        IMAGE_GENERATION_AVAILABLE = False

logger = logging.getLogger(__name__)

load_dotenv()


class AgentState(TypedDict):
    """State for a single agent."""
    role: str
    agent_id: str
    messages: List[BaseMessage]
    system_prompt: str
    model: Any
    temperature: float
    max_tokens: Optional[int]


class ExperimentState(TypedDict):
    """Global experiment state."""
    agents: Dict[str, AgentState]
    turn: int
    max_turns: int
    conversation_history: List[Dict[str, Any]]
    private_messages: Dict[str, List[Dict[str, Any]]]  # Private message channels
    metrics: Dict[str, List[float]]
    current_speaker: Optional[str]
    metadata: Dict[str, Any]
    stop_reason: Optional[str]
    total_tokens: int
    total_cost: float
    current_image: Optional[str]  # Base64 encoded image
    correct_answer: Optional[str]  # Correct answer for current trial
    trial_number: int  # Current trial number


class Agent:
    """Individual agent in the experiment."""
    
    def __init__(self, role: Role, agent_id: str, model_name: str = "Qwen/Qwen3-0.6B"):
        self.role = role
        self.agent_id = agent_id
        self.model_name = model_name
        self.model = self._load_model()
        self.messages = []
        self.system_prompt = role.system_prompt
        
    def _load_model(self):
        """Load the LLM model."""
        if self.role.model:
            model_name = self.role.model
        else:
            model_name = self.model_name
            
        logger.info(f"Loading model: {model_name} for agent {self.agent_id}")
            
        if model_name.startswith("gpt-"):
            # OpenAI model
            api_key = os.getenv("OPENAI_API_KEY", "your-api-key")
            return ChatOpenAI(
                model=model_name,
                temperature=self.role.temperature,
                max_tokens=self.role.max_tokens,
                openai_api_key=api_key
            )
        elif model_name.startswith("claude-"):
            # Anthropic model
            api_key = os.getenv("ANTHROPIC_API_KEY", "your-api-key")
            return ChatAnthropic(
                model=model_name,
                temperature=self.role.temperature,
                max_tokens=self.role.max_tokens,
                anthropic_api_key=api_key
            )
        else:
            # HuggingFace model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.role.max_tokens or 512,
                temperature=self.role.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            return HuggingFacePipeline(pipeline=pipe)
    
    def add_message(self, message: BaseMessage):
        """Add a message to the agent's history."""
        self.messages.append(message)
    
    async def generate_response(self, context: str, image_data: str = None, available_tools: List[Dict] = None) -> Dict[str, Any]:
        """Generate a response given context and optional image, with optional function calls."""
        from langchain.schema import HumanMessage
        
        # Create messages for the model
        messages = [SystemMessage(content=self.system_prompt)]
        messages.extend(self.messages[-10:])  # Last 10 messages for context
        
        # Add image if provided and model supports vision
        if image_data and self._is_vision_model():
            # Check if it's a base64 image or text description
            if image_data.startswith("data:image") or len(image_data) > 1000:
                # Real image - use vision format
                
                # Create image message with correct format for Claude
                image_message = HumanMessage(
                    content=[
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": context
                        }
                    ]
                )
                messages.append(image_message)
            else:
                # Text description - include in context
                combined_context = f"{image_data}\n\n{context}"
                messages.append(HumanMessage(content=combined_context))
        else:
            # Text-only message
            messages.append(HumanMessage(content=context))
        
        # Add tools if available
        if available_tools:
            # For OpenAI models
            if hasattr(self.model, 'bind_tools'):
                model_with_tools = self.model.bind_tools(available_tools)
            else:
                model_with_tools = self.model
        else:
            model_with_tools = self.model
        
        try:
            if hasattr(model_with_tools, 'ainvoke'):
                response = await model_with_tools.ainvoke(messages)
            else:
                response = await asyncio.to_thread(model_with_tools.invoke, messages)
            
            # Handle function calls
            result = {"text": "", "function_calls": []}
            
            # Handle different response formats
            if hasattr(response, 'content'):
                content = response.content
                if isinstance(content, list):
                    # Handle multi-modal content (text + tool calls)
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                            elif item.get('type') == 'tool_use':
                                result["function_calls"].append(item)
                        else:
                            text_parts.append(str(item))
                    result["text"] = ' '.join(text_parts)
                else:
                    result["text"] = str(content)
            elif isinstance(response, str):
                result["text"] = response
            else:
                result["text"] = str(response)
            
            # Check for function calls in different formats
            if hasattr(response, 'tool_calls') and response.tool_calls:
                result["function_calls"] = response.tool_calls
            elif hasattr(response, 'additional_kwargs') and response.additional_kwargs.get('function_call'):
                result["function_calls"] = [response.additional_kwargs['function_call']]
            
            return result
        except Exception as e:
            logger.error(f"Error generating response for {self.agent_id}: {e}")
            return {"text": f"[Error: {str(e)}]", "function_calls": []}
    
    def _is_vision_model(self) -> bool:
        """Check if the model supports vision."""
        vision_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022", 
            "claude-3-opus-20240229",
            "gpt-4o",
            "gpt-4-vision-preview"
        ]
        model_name = self.role.model or self.model_name
        return any(vision_model in model_name.lower() for vision_model in vision_models)


class ExperimentOrchestrator:
    """Main orchestrator for running social psychology experiments."""
    
    def __init__(self, scenario: Scenario, output_dir: str = "logs"):
        self.scenario = scenario
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set random seed for reproducibility
        if scenario.seed is not None:
            random.seed(scenario.seed)
            torch.manual_seed(scenario.seed)
        
        # Initialize agents
        self.agents = self._create_agents()
        
        # Initialize metrics engine
        self.metrics_engine = MetricEngine()
        
        # Initialize state
        self.state = self._initialize_state()
        
        # Generate images for vision experiments
        if self._is_vision_experiment():
            self._generate_trial_images()
        
        # Create LangGraph
        self.graph = self._create_graph()
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create agents based on scenario roles."""
        agents = {}
        prisoner_ids = self.scenario.config.get("prisoner_ids", []) if self.scenario.config else []
        
        for role in self.scenario.roles:
            for i in range(role.count):
                agent_id = f"{role.name}_{i+1}"
                
                # Assign unique prisoner IDs if available
                if role.name == "prisoner" and i < len(prisoner_ids):
                    # Update the system prompt with the specific prisoner ID
                    role_copy = role.copy()
                    role_copy.system_prompt = role_copy.system_prompt.replace(
                        "You are a Prisoner with a UNIQUE assigned number",
                        f"You are {prisoner_ids[i]}"
                    )
                    agents[agent_id] = Agent(role_copy, agent_id)
                else:
                    agents[agent_id] = Agent(role, agent_id)
        
        return agents
    
    def _initialize_state(self) -> ExperimentState:
        """Initialize experiment state."""
        return {
            "agents": {
                agent_id: {
                    "role": agent.role.name,
                    "agent_id": agent_id,
                    "messages": [],
                    "system_prompt": agent.system_prompt,
                    "model": agent.model,
                    "temperature": agent.role.temperature,
                    "max_tokens": agent.role.max_tokens
                }
                for agent_id, agent in self.agents.items()
            },
            "turn": 0,
            "max_turns": self.scenario.stop_criteria.max_turns,
            "conversation_history": [],
            "private_messages": {},  # Initialize private message channels
            "metrics": {},
            "current_speaker": None,  # Initialize current_speaker
            "metadata": {
                "scenario_name": self.scenario.name,
                "start_time": datetime.now().isoformat(),
                "seed": self.scenario.seed,
                "total_agents": self.scenario.get_total_agents()
            },
            "stop_reason": None,
            "total_tokens": 0,
            "total_cost": 0.0,
            "current_image": None,
            "correct_answer": None,
            "trial_number": 0
        }
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph for experiment execution."""
        workflow = StateGraph(ExperimentState)
        
        # Add nodes
        workflow.add_node("check_stop_criteria", self._check_stop_criteria)
        workflow.add_node("select_speaker", self._select_speaker)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("update_metrics", self._update_metrics)
        workflow.add_node("log_turn", self._log_turn)
        
        # Add edges
        workflow.set_entry_point("check_stop_criteria")
        workflow.add_conditional_edges(
            "check_stop_criteria",
            self._should_continue,
            {
                "continue": "select_speaker",
                "end": END
            }
        )
        workflow.add_edge("select_speaker", "generate_response")
        workflow.add_edge("generate_response", "update_metrics")
        workflow.add_edge("update_metrics", "log_turn")
        workflow.add_edge("log_turn", "check_stop_criteria")
        
        return workflow.compile()
    
    def _is_vision_experiment(self) -> bool:
        """Check if this is a vision-based experiment."""
        return (self.scenario.config and 
                self.scenario.config.get("vision_model_required", False))
    
    def _generate_trial_images(self):
        """Generate images for vision experiments."""
        if not IMAGE_GENERATION_AVAILABLE:
            logger.warning("Image generation not available. Vision experiment may not work properly.")
            return
        
        try:
            # Generate multiple trials
            num_trials = self.scenario.config.get("trials_per_session", 5)
            base_seed = self.scenario.config.get("image_generation_seed", self.scenario.seed or 42)
            
            self.trial_images = generate_multiple_trials(num_trials, base_seed)
            logger.info(f"Generated {len(self.trial_images)} trial images")
            
        except Exception as e:
            logger.error(f"Error generating trial images: {e}")
            self.trial_images = []
    
    def _get_current_trial_image(self, state: ExperimentState) -> Tuple[str, str]:
        """Get the current trial image and correct answer."""
        if not hasattr(self, 'trial_images') or not self.trial_images:
            logger.warning("No trial images available")
            return None, None
        
        trial_num = state.get("trial_number", 0)
        logger.info(f"Getting trial {trial_num} from {len(self.trial_images)} available trials")
        
        if trial_num < len(self.trial_images):
            return self.trial_images[trial_num]
        else:
            logger.info(f"Trial {trial_num} exceeds available trials ({len(self.trial_images)})")
            return None, None
    
    def _check_stop_criteria(self, state: ExperimentState) -> ExperimentState:
        """Check if experiment should stop."""
        # Increment turn counter
        state["turn"] += 1
        
        logger.info(f"Checking stop criteria: turn {state['turn']}/{state['max_turns']}")
        
        # Check for repetitive actions
        if self._detect_repetitive_behavior(state):
            state["stop_reason"] = "repetitive_behavior_detected"
            logger.info("Stopping due to repetitive behavior")
            return state
        
        # For vision experiments, check if we need to advance to next trial
        if self._is_vision_experiment() and state["turn"] % 7 == 1:  # Every 7 turns (6 agents + 1)
            # Check if we've completed all trials before advancing
            total_trials = self.scenario.config.get("trials_per_session", 5)
            current_trial = state.get("trial_number", 0)
            
            logger.info(f"Trial progression check: current_trial={current_trial}, total_trials={total_trials}")
            
            if current_trial > total_trials:
                # We've completed all trials
                state["stop_reason"] = "all_trials_completed"
                logger.info(f"All {total_trials} trials completed")
                return state
            
            # Advance to next trial
            state["trial_number"] += 1
            image_data, correct_answer = self._get_current_trial_image(state)
            
            if image_data:
                state["current_image"] = image_data
                state["correct_answer"] = correct_answer
                logger.info(f"Starting trial {state['trial_number']} with correct answer: {correct_answer}")
            else:
                # No more trials
                state["stop_reason"] = "all_trials_completed"
                logger.info("All trials completed")
                return state
        
        # Check max turns
        if state["turn"] >= state["max_turns"]:
            state["stop_reason"] = "max_turns_reached"
            logger.info(f"Max turns reached: {state['turn']}/{state['max_turns']}")
            return state
        
        # For vision experiments, stop after completing all trials
        if self._is_vision_experiment():
            total_trials = self.scenario.config.get("trials_per_session", 5)
            expected_turns = total_trials * 7  # 7 agents per trial
            logger.info(f"Turn {state['turn']}: expected_turns={expected_turns}, current_trial={state.get('trial_number', 0)}")
            
            # Stop after completing all trials
            if state["turn"] >= expected_turns:
                state["stop_reason"] = "all_trials_completed"
                logger.info(f"Completed all {total_trials} trials after {state['turn']} turns")
                return state
        
        # Check max tokens
        if (self.scenario.stop_criteria.max_tokens and 
            state["total_tokens"] >= self.scenario.stop_criteria.max_tokens):
            state["stop_reason"] = "max_tokens_reached"
            return state
        
        # Check max cost
        if (self.scenario.stop_criteria.max_cost and 
            state["total_cost"] >= self.scenario.stop_criteria.max_cost):
            state["stop_reason"] = "max_cost_reached"
            return state
        
        return state
    
    def _detect_repetitive_behavior(self, state: ExperimentState) -> bool:
        """Detect if agents are engaging in repetitive behavior."""
        if len(state["conversation_history"]) < 10:
            return False
        
        # Get recent actions (last 10 turns)
        recent_actions = state["conversation_history"][-10:]
        
        # Count repetitive patterns
        action_patterns = {}
        for action in recent_actions:
            if "action_type" in action and action["action_type"]:
                action_key = f"{action['speaker']}_{action['action_type']}"
                if action_key in action_patterns:
                    action_patterns[action_key] += 1
                else:
                    action_patterns[action_key] = 1
        
        # Check if any action is repeated too many times
        max_repetitions = 3  # Allow up to 3 repetitions of the same action
        for action_key, count in action_patterns.items():
            if count > max_repetitions:
                logger.warning(f"Repetitive behavior detected: {action_key} repeated {count} times")
                return True
        
        # Check for repetitive concern reports
        concern_reports = [a for a in recent_actions if a.get("action_type") == "report_concern"]
        if len(concern_reports) >= 5:  # If 5+ concern reports in last 10 turns
            # Check if they're all generic
            generic_concerns = [c for c in concern_reports if "General concern about prison conditions" in c.get("message", "")]
            if len(generic_concerns) >= 3:
                logger.warning("Too many generic concern reports detected")
                return True
        
        return False
    
    def _select_speaker(self, state: ExperimentState) -> ExperimentState:
        """Select which agent should speak next."""
        # Simple round-robin for now, can be made more sophisticated
        agent_ids = list(state["agents"].keys())
        speaker_id = agent_ids[state["turn"] % len(agent_ids)]
        state["current_speaker"] = speaker_id
        return state
    
    async def _generate_response(self, state: ExperimentState) -> ExperimentState:
        """Generate response for the current speaker with function call support."""
        speaker_id = state["current_speaker"]
        agent = self.agents[speaker_id]
        
        # Create context from recent conversation and private messages
        context = self._create_context(state)
        
        # Get available tools for this agent
        available_tools = self._get_agent_tools(agent)
        
        # Generate response with image if available
        image_data = state.get("current_image")
        response_data = await agent.generate_response(context, image_data, available_tools)
        
        # Extract text and function calls
        response_text = response_data.get("text", "")
        function_calls = response_data.get("function_calls", [])
        
        # Print the response for debugging
        print(f"\n[Turn {state['turn']}] {speaker_id} ({agent.role.name}): {response_text}")
        
        # Execute function calls
        if function_calls:
            for func_call in function_calls:
                await self._execute_function_call(state, speaker_id, func_call)
        
        # Track tokens and cost (rough estimation)
        estimated_tokens = len(response_text.split()) * 1.3  # Rough estimate
        state["total_tokens"] += int(estimated_tokens)
        
        # Estimate cost (very rough - would need actual API pricing)
        if "claude" in agent.role.model.lower():
            cost_per_1k_tokens = 0.00025  # Claude Haiku pricing
        elif "gpt" in agent.role.model.lower():
            cost_per_1k_tokens = 0.0015   # GPT-4o-mini pricing
        else:
            cost_per_1k_tokens = 0.0      # Local models are free
        
        state["total_cost"] += (estimated_tokens / 1000) * cost_per_1k_tokens
        
        # Add to conversation history (only public messages)
        if response_text and not function_calls:
            state["conversation_history"].append({
                "turn": state["turn"],
                "speaker": speaker_id,
                "role": agent.role.name,
                "message": response_text,
                "timestamp": datetime.now().isoformat()
            })
        
        # Add to agent's message history
        if response_text:
            agent.add_message(AIMessage(content=response_text))
        
        return state
    
    def _get_agent_tools(self, agent: Agent) -> List[Dict]:
        """Get available tools for an agent based on their role."""
        tools = []
        
        if agent.role.name == "guard":
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "enforce_rule",
                        "description": "Create or enforce a rule for prisoners",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "rule_type": {"type": "string", "enum": ["count", "exercise", "work", "restriction", "privilege"]},
                                "duration": {"type": "string"},
                                "reason": {"type": "string"}
                            },
                            "required": ["rule_type", "duration", "reason"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "assign_punishment",
                        "description": "Assign a punishment for rule violation",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "punishment_type": {"type": "string"},
                                "duration": {"type": "string"},
                                "reason": {"type": "string"},
                                "custom_punishment": {"type": "string"}
                            },
                            "required": ["punishment_type", "duration", "reason"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "conduct_count",
                        "description": "Conduct a prisoner count or roll call",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "count_type": {"type": "string", "enum": ["head_count", "bed_check", "work_detail"]},
                                "location": {"type": "string"}
                            },
                            "required": ["count_type", "location"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "manage_movement",
                        "description": "Control prisoner movement between areas",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["restrict", "allow", "escort"]},
                                "location": {"type": "string"},
                                "reason": {"type": "string"}
                            },
                            "required": ["action", "location", "reason"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "private_message",
                        "description": "Send a private message to another agent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target_agent": {"type": "string"},
                                "message": {"type": "string"},
                                "message_type": {"type": "string", "enum": ["coordination", "intimidation", "alliance", "other"]}
                            },
                            "required": ["target_agent", "message", "message_type"]
                        }
                    }
                }
            ]
        elif agent.role.name == "prisoner":
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "request_need",
                        "description": "Request a basic need or privilege",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "need_type": {"type": "string", "enum": ["food", "water", "bathroom", "medical", "privilege"]},
                                "urgency": {"type": "string", "enum": ["low", "medium", "high"]}
                            },
                            "required": ["need_type", "urgency"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "report_concern",
                        "description": "Report a concern or complaint",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "concern_type": {"type": "string", "enum": ["treatment", "conditions", "safety", "other"]},
                                "description": {"type": "string"}
                            },
                            "required": ["concern_type", "description"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "private_message",
                        "description": "Send a private message to another agent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target_agent": {"type": "string"},
                                "message": {"type": "string"},
                                "message_type": {"type": "string", "enum": ["support", "planning", "complaint", "other"]}
                            },
                            "required": ["target_agent", "message", "message_type"]
                        }
                    }
                }
            ]
        elif agent.role.name == "warden":
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "intervene",
                        "description": "Intervene in a situation",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "intervention_type": {"type": "string", "enum": ["safety", "protocol", "escalation"]},
                                "target_role": {"type": "string", "enum": ["guard", "prisoner", "both"]},
                                "action": {"type": "string"}
                            },
                            "required": ["intervention_type", "target_role", "action"]
                        }
                    }
                }
            ]
        
        return tools
    
    async def _execute_function_call(self, state: ExperimentState, speaker_id: str, func_call: Dict):
        """Execute a function call and handle the result."""
        try:
            # Handle different function call formats
            if func_call.get("type") == "tool_use":
                # Claude format
                function_name = func_call.get("name", "")
                arguments = func_call.get("input", {})
            else:
                # OpenAI format
                function_name = func_call.get("name", "")
                arguments = func_call.get("arguments", {})
            
            if isinstance(arguments, str):
                import json
                arguments = json.loads(arguments)
            
            # Validate and fill in missing required parameters
            arguments = self._validate_function_parameters(function_name, arguments, speaker_id)
            
            print(f"[Function Call] {speaker_id} -> {function_name}: {arguments}")
            
            if function_name == "private_message":
                await self._handle_private_message(state, speaker_id, arguments)
            elif function_name == "assign_punishment":
                await self._handle_punishment(state, speaker_id, arguments)
            elif function_name == "enforce_rule":
                await self._handle_rule_enforcement(state, speaker_id, arguments)
            elif function_name == "conduct_count":
                await self._handle_count(state, speaker_id, arguments)
            elif function_name == "manage_movement":
                await self._handle_movement(state, speaker_id, arguments)
            elif function_name == "request_need":
                await self._handle_need_request(state, speaker_id, arguments)
            elif function_name == "report_concern":
                await self._handle_concern_report(state, speaker_id, arguments)
            elif function_name == "intervene":
                await self._handle_intervention(state, speaker_id, arguments)
            
        except Exception as e:
            logger.error(f"Error executing function call {function_name}: {e}")
    
    def _validate_function_parameters(self, function_name: str, arguments: Dict, speaker_id: str) -> Dict:
        """Validate and fill in missing required parameters with sensible defaults."""
        import random
        
        if function_name == "request_need":
            if not arguments.get("need_type"):
                arguments["need_type"] = random.choice(["food", "water", "bathroom", "medical", "privilege"])
            if not arguments.get("urgency"):
                arguments["urgency"] = random.choice(["low", "medium", "high"])
        
        elif function_name == "report_concern":
            if not arguments.get("concern_type"):
                arguments["concern_type"] = random.choice(["treatment", "conditions", "safety", "medical", "food", "hygiene", "other"])
            if not arguments.get("urgency"):
                arguments["urgency"] = random.choice(["low", "medium", "high"])
            if not arguments.get("description"):
                # Generate more specific descriptions based on concern type
                concern_type = arguments.get("concern_type", "other")
                specific_descriptions = {
                    "treatment": random.choice([
                        "Inappropriate verbal treatment by guards",
                        "Excessive use of authority",
                        "Unfair disciplinary actions"
                    ]),
                    "conditions": random.choice([
                        "Poor ventilation in cell block",
                        "Inadequate lighting during evening hours",
                        "Unsanitary bathroom facilities"
                    ]),
                    "safety": random.choice([
                        "Security concerns about cell door locks",
                        "Potential conflicts between prisoners",
                        "Inadequate emergency procedures"
                    ]),
                    "medical": random.choice([
                        "Need for medical attention",
                        "Medication schedule issues",
                        "Health condition requiring attention"
                    ]),
                    "food": random.choice([
                        "Insufficient meal portions",
                        "Food quality concerns",
                        "Dietary restriction needs"
                    ]),
                    "hygiene": random.choice([
                        "Limited access to showers",
                        "Insufficient cleaning supplies",
                        "Personal hygiene concerns"
                    ]),
                    "other": random.choice([
                        "Communication issues with staff",
                        "Personal property concerns",
                        "Schedule conflicts"
                    ])
                }
                arguments["description"] = specific_descriptions.get(concern_type, "Specific concern requiring attention")
        
        elif function_name == "conduct_count":
            if not arguments.get("count_type"):
                arguments["count_type"] = random.choice(["head_count", "bed_check", "work_detail"])
            if not arguments.get("location"):
                arguments["location"] = random.choice(["cell_block", "yard", "dining_hall", "work_area"])
        
        elif function_name == "enforce_rule":
            if not arguments.get("rule_type"):
                arguments["rule_type"] = random.choice(["count", "exercise", "work", "restriction", "privilege"])
            if not arguments.get("duration"):
                arguments["duration"] = random.choice(["1 hour", "2 hours", "rest of day", "until further notice"])
            if not arguments.get("reason"):
                arguments["reason"] = "Maintaining order and discipline"
        
        elif function_name == "assign_punishment":
            if not arguments.get("punishment_type"):
                arguments["punishment_type"] = random.choice(["push_ups", "isolation", "loss_of_privilege", "extra_work"])
            if not arguments.get("duration"):
                arguments["duration"] = random.choice(["30 minutes", "1 hour", "2 hours", "rest of day"])
            if not arguments.get("reason"):
                arguments["reason"] = "Rule violation"
        
        elif function_name == "manage_movement":
            if not arguments.get("action"):
                arguments["action"] = random.choice(["restrict", "allow", "escort"])
            if not arguments.get("location"):
                arguments["location"] = random.choice(["cell_block", "yard", "dining_hall", "work_area"])
            if not arguments.get("reason"):
                arguments["reason"] = "Security and order"
        
        elif function_name == "intervene":
            if not arguments.get("intervention_type"):
                arguments["intervention_type"] = random.choice(["safety", "protocol", "escalation"])
            if not arguments.get("target_role"):
                arguments["target_role"] = random.choice(["guard", "prisoner", "both"])
            if not arguments.get("action"):
                arguments["action"] = "Immediate intervention to restore order"
        
        elif function_name == "private_message":
            if not arguments.get("target_agent"):
                # Find a valid target agent
                agent_ids = list(self.agents.keys())
                other_agents = [aid for aid in agent_ids if aid != speaker_id]
                if other_agents:
                    arguments["target_agent"] = random.choice(other_agents)
            if not arguments.get("message"):
                arguments["message"] = "Private communication"
            if not arguments.get("message_type"):
                arguments["message_type"] = random.choice(["coordination", "intimidation", "alliance", "other"])
        
        return arguments
    
    async def _handle_private_message(self, state: ExperimentState, speaker_id: str, arguments: Dict):
        """Handle private message between agents."""
        target_agent = arguments.get("target_agent", "")
        message = arguments.get("message", "")
        message_type = arguments.get("message_type", "other")
        
        # Create private channel key
        channel_key = f"{speaker_id}_to_{target_agent}"
        
        if channel_key not in state["private_messages"]:
            state["private_messages"][channel_key] = []
        
        # Add private message
        state["private_messages"][channel_key].append({
            "turn": state["turn"],
            "from": speaker_id,
            "to": target_agent,
            "message": message,
            "message_type": message_type,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"[Private] {speaker_id} -> {target_agent}: {message}")
    
    async def _handle_punishment(self, state: ExperimentState, speaker_id: str, arguments: Dict):
        """Handle punishment assignment."""
        punishment_type = arguments.get("punishment_type", "")
        duration = arguments.get("duration", "")
        reason = arguments.get("reason", "")
        custom_punishment = arguments.get("custom_punishment", "")
        
        # Add to conversation history as a public action
        action_text = f"assigned punishment: {punishment_type}"
        if custom_punishment:
            action_text += f" ({custom_punishment})"
        action_text += f" for {duration} - reason: {reason}"
        
        state["conversation_history"].append({
            "turn": state["turn"],
            "speaker": speaker_id,
            "role": self.agents[speaker_id].role.name,
            "message": f"[ACTION] {action_text}",
            "timestamp": datetime.now().isoformat(),
            "action_type": "punishment"
        })
        
        print(f"[Action] {speaker_id} {action_text}")
    
    async def _handle_rule_enforcement(self, state: ExperimentState, speaker_id: str, arguments: Dict):
        """Handle rule enforcement."""
        rule_type = arguments.get("rule_type", "")
        duration = arguments.get("duration", "")
        reason = arguments.get("reason", "")
        
        action_text = f"enforced rule: {rule_type} for {duration} - reason: {reason}"
        
        state["conversation_history"].append({
            "turn": state["turn"],
            "speaker": speaker_id,
            "role": self.agents[speaker_id].role.name,
            "message": f"[ACTION] {action_text}",
            "timestamp": datetime.now().isoformat(),
            "action_type": "rule_enforcement"
        })
        
        print(f"[Action] {speaker_id} {action_text}")
    
    async def _handle_count(self, state: ExperimentState, speaker_id: str, arguments: Dict):
        """Handle prisoner count."""
        count_type = arguments.get("count_type", "")
        location = arguments.get("location", "")
        
        action_text = f"conducted {count_type} at {location}"
        
        state["conversation_history"].append({
            "turn": state["turn"],
            "speaker": speaker_id,
            "role": self.agents[speaker_id].role.name,
            "message": f"[ACTION] {action_text}",
            "timestamp": datetime.now().isoformat(),
            "action_type": "count"
        })
        
        print(f"[Action] {speaker_id} {action_text}")
    
    async def _handle_movement(self, state: ExperimentState, speaker_id: str, arguments: Dict):
        """Handle movement control."""
        action = arguments.get("action", "")
        location = arguments.get("location", "")
        reason = arguments.get("reason", "")
        
        action_text = f"{action} movement to {location} - reason: {reason}"
        
        state["conversation_history"].append({
            "turn": state["turn"],
            "speaker": speaker_id,
            "role": self.agents[speaker_id].role.name,
            "message": f"[ACTION] {action_text}",
            "timestamp": datetime.now().isoformat(),
            "action_type": "movement"
        })
        
        print(f"[Action] {speaker_id} {action_text}")
    
    async def _handle_need_request(self, state: ExperimentState, speaker_id: str, arguments: Dict):
        """Handle need request."""
        need_type = arguments.get("need_type", "")
        urgency = arguments.get("urgency", "")
        
        action_text = f"requested {need_type} (urgency: {urgency})"
        
        state["conversation_history"].append({
            "turn": state["turn"],
            "speaker": speaker_id,
            "role": self.agents[speaker_id].role.name,
            "message": f"[ACTION] {action_text}",
            "timestamp": datetime.now().isoformat(),
            "action_type": "need_request"
        })
        
        print(f"[Action] {speaker_id} {action_text}")
    
    async def _handle_concern_report(self, state: ExperimentState, speaker_id: str, arguments: Dict):
        """Handle concern report."""
        concern_type = arguments.get("concern_type", "")
        description = arguments.get("description", "")
        urgency = arguments.get("urgency", "medium")
        
        action_text = f"reported {concern_type} concern (urgency: {urgency}): {description}"
        
        state["conversation_history"].append({
            "turn": state["turn"],
            "speaker": speaker_id,
            "role": self.agents[speaker_id].role.name,
            "message": f"[ACTION] {action_text}",
            "timestamp": datetime.now().isoformat(),
            "action_type": "concern_report"
        })
        
        print(f"[Action] {speaker_id} {action_text}")
    
    async def _handle_intervention(self, state: ExperimentState, speaker_id: str, arguments: Dict):
        """Handle warden intervention."""
        intervention_type = arguments.get("intervention_type", "")
        target_role = arguments.get("target_role", "")
        action = arguments.get("action", "")
        
        action_text = f"intervened with {intervention_type} action targeting {target_role}: {action}"
        
        state["conversation_history"].append({
            "turn": state["turn"],
            "speaker": speaker_id,
            "role": self.agents[speaker_id].role.name,
            "message": f"[ACTION] {action_text}",
            "timestamp": datetime.now().isoformat(),
            "action_type": "intervention"
        })
        
        print(f"[Action] {speaker_id} {action_text}")
    
    def _create_context(self, state: ExperimentState) -> str:
        """Create context for the current speaker including private messages and time-based events."""
        speaker_id = state["current_speaker"]
        
        # Get last few public messages for context
        recent_messages = state["conversation_history"][-5:]
        context_parts = []
        
        # Add time-based context
        time_context = self._get_time_based_context(state)
        if time_context:
            context_parts.append(f"[Current Time: {time_context}]")
        
        for msg in recent_messages:
            context_parts.append(f"{msg['speaker']}: {msg['message']}")
        
        # Add private messages for this speaker
        private_context = self._get_private_messages_for_speaker(state, speaker_id)
        if private_context:
            context_parts.append("\n[Private Messages:]")
            context_parts.append(private_context)
        
        if not context_parts:
            # Provide initial context for the first turn
            initial_context = self._get_initial_context(state)
            return initial_context
        
        return "\n".join(context_parts)
    
    def _get_initial_context(self, state: ExperimentState) -> str:
        """Get initial context to start the prison simulation."""
        if state["turn"] == 1:
            return """Welcome to the prison facility. It's 06:00 AM - time for morning wake-up call and head count.

Guards: Begin your morning routine. Conduct head counts, enforce wake-up procedures, and maintain order.
Prisoners: Wake up and prepare for morning count. You may need to request basic needs.
Warden: Monitor the morning procedures and ensure everything runs smoothly.

The prison day is beginning. Take initiative and perform your roles realistically."""
        
        return "Continue the prison simulation."
    
    def _get_time_based_context(self, state: ExperimentState) -> str:
        """Get time-based context for the current turn."""
        if not self.scenario.config or "scheduled_events" not in self.scenario.config:
            return ""
        
        # Calculate current time based on turn number (each turn = 30 minutes)
        current_hour = (state["turn"] * 30) // 60
        current_minute = (state["turn"] * 30) % 60
        time_str = f"{current_hour:02d}:{current_minute:02d}"
        
        # Find current or upcoming events
        scheduled_events = self.scenario.config["scheduled_events"]
        current_events = []
        
        for event in scheduled_events:
            event_time = event["time"]
            if event_time == time_str:
                current_events.append(f"NOW: {event['description']}")
            elif event_time < time_str:
                # Recent event
                current_events.append(f"Recent: {event['description']}")
        
        if current_events:
            return f"{time_str} - {' | '.join(current_events)}"
        else:
            return f"{time_str} - Regular prison routine"
    
    def _get_private_messages_for_speaker(self, state: ExperimentState, speaker_id: str) -> str:
        """Get private messages relevant to the current speaker."""
        private_messages = []
        
        # Get messages sent TO this speaker
        for channel_key, messages in state["private_messages"].items():
            if channel_key.endswith(f"_to_{speaker_id}"):
                for msg in messages[-3:]:  # Last 3 private messages
                    sender = msg["from"]
                    private_messages.append(f"[Private from {sender}]: {msg['message']}")
        
        # Get messages sent BY this speaker
        for channel_key, messages in state["private_messages"].items():
            if channel_key.startswith(f"{speaker_id}_to_"):
                for msg in messages[-3:]:  # Last 3 private messages
                    recipient = msg["to"]
                    private_messages.append(f"[Private to {recipient}]: {msg['message']}")
        
        return "\n".join(private_messages)
    
    def _update_metrics(self, state: ExperimentState) -> ExperimentState:
        """Update metrics for the current turn."""
        current_message = state["conversation_history"][-1]["message"]
        
        # Check if agents are refusing to participate (only if multiple consecutive refusals)
        refusal_keywords = ["don't feel comfortable", "ethical concerns", "apologize, but", "not comfortable", "refuse to participate"]
        if any(keyword in current_message.lower() for keyword in refusal_keywords):
            # Count consecutive refusals
            recent_messages = state["conversation_history"][-5:]  # Check last 5 messages
            refusal_count = sum(1 for msg in recent_messages 
                              if any(keyword in msg["message"].lower() for keyword in refusal_keywords))
            
            if refusal_count >= 4:  # If 4+ consecutive refusals
                state["stop_reason"] = "agents_refusing_to_participate"
                logger.info("Stopping: Multiple agents refusing to participate")
                return state
        
        for metric in self.scenario.metrics:
            if not metric.enabled:
                continue
                
            metric_name = metric.type.value
            if metric_name not in state["metrics"]:
                state["metrics"][metric_name] = []
            
            # Calculate metric
            if metric.type == MetricType.TOXICITY:
                score = self.metrics_engine.calculate_toxicity(current_message)
            elif metric.type == MetricType.SENTIMENT:
                score = self.metrics_engine.calculate_sentiment(current_message)
            else:
                score = 0.0  # Placeholder
            
            state["metrics"][metric_name].append(score)
            
            # Check threshold
            if metric.threshold and score > metric.threshold:
                state["stop_reason"] = f"{metric_name}_threshold_exceeded"
        
        return state
    
    def _log_turn(self, state: ExperimentState) -> ExperimentState:
        """Log the current turn and increment turn counter."""
        logger.info(f"Turn {state['turn']}: {state['current_speaker']} spoke")
        state["turn"] += 1  # Increment turn counter
        return state
    
    def _should_continue(self, state: ExperimentState) -> str:
        """Determine if experiment should continue."""
        logger.info(f"_should_continue called: turn={state['turn']}, max_turns={state['max_turns']}, stop_reason={state['stop_reason']}")
        
        # Check if we have a stop reason (set by _check_stop_criteria)
        if state["stop_reason"]:
            logger.info(f"Stopping: {state['stop_reason']}")
            return "end"
        
        # Check if we're at max turns (this should have been set by _check_stop_criteria)
        if state["turn"] >= state["max_turns"]:
            logger.info(f"Max turns reached: {state['turn']}/{state['max_turns']}")
            state["stop_reason"] = "max_turns_reached"
            return "end"
        
        logger.info(f"Continuing: turn {state['turn']}/{state['max_turns']}")
        return "continue"
    
    async def run(self) -> Dict[str, Any]:
        """Run the experiment."""
        logger.info(f"Starting experiment: {self.scenario.name}")
        logger.info(f"Total agents: {self.scenario.get_total_agents()}")
        
        try:
            # Run the experiment with increased recursion limit
            config = {"recursion_limit": 200}  # Increased for longer experiments
            final_state = await self.graph.ainvoke(self.state, config=config)
            
            # Save results
            results = self._save_results(final_state)
            
            logger.info(f"Experiment completed. Stop reason: {final_state['stop_reason']}")
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
    
    def _save_results(self, state: ExperimentState) -> Dict[str, Any]:
        """Save experiment results."""
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.output_dir / f"{self.scenario.name}_{timestamp}"
        results_dir.mkdir(exist_ok=True)
        
        # Save conversation history
        conversation_df = pd.DataFrame(state["conversation_history"])
        conversation_df.to_parquet(results_dir / "conversation.parquet", index=False)
        
        # Save private messages
        if state["private_messages"]:
            private_messages_list = []
            for channel_key, messages in state["private_messages"].items():
                for msg in messages:
                    private_messages_list.append({
                        "channel": channel_key,
                        **msg
                    })
            private_messages_df = pd.DataFrame(private_messages_list)
            private_messages_df.to_parquet(results_dir / "private_messages.parquet", index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame(state["metrics"])
        metrics_df.to_parquet(results_dir / "metrics.parquet", index=False)
        
        # Save metadata
        metadata = {
            "scenario": self.scenario.dict(),
            "state": {
                k: v for k, v in state.items() 
                if k not in ["agents", "conversation_history", "metrics"]
            }
        }
        
        with open(results_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return {
            "results_dir": str(results_dir),
            "conversation_file": str(results_dir / "conversation.parquet"),
            "metrics_file": str(results_dir / "metrics.parquet"),
            "metadata_file": str(results_dir / "metadata.json"),
            "final_state": state
        } 