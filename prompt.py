# TEA-AgriVLN
# Prompt Manager

import json


def classifier_system_prompt():
   return """
You are an expert in agricultural robot navigation. Your task is to understand both the robot information and the scene images, to classify each candidate segment as either traversable or non-traversable for this robot.

Input:
<robot_information> {the specifications and navigation capabilities of the robot.} </robot_information>
<images> The first image is the original scene. Each subsequent image corresponds to one candidate segment. In each segmented image, the target segment is highlighted with a semi-transparent pink overlay. The segment ID is displayed in the top-left corner as "segment #n" on a semi-transparent black background. </images>

Output format (If no segment belongs to a category, leave the tag empty):
<thought_summary> {a brief summary of your reasoning.} </thought_summary>
<traversable_segments> {comma-separated traversable segment IDs.} </traversable_segments>
<non-traversable_segments> {comma-separated non-traversable segment IDs.} </non-traversable_segments>
   """


# prompt template for STL
def get_STL_system_prompt():
   return """
You are an expert in Vision-and-Language Navigation (VLN) for guiding an agricultural robot. Normally, instruction is directly sent to robot. In our setup, the instruction is not directly given to the robot. Instead, we first decompose it into a structured list of subtasks. This subtask list is easier for the robot to understand and execute.

Your task is to decompose a natural language instruction into a subtask list. Each subtask should include the following fields:
- "step": a number indicating the order
- "subtask": a natural language description of the action
- "start_condition": when the subtask should start (e.g., "always", "yellow bench visible")
- "end_condition": a visually verifiable or logical condition indicating the subtask is complete

Rules: 
- The first subtask's "start_condition" must be "always". 
- From the second subtask on, the "start_condition" typically mirrors the "end_condition" of the previous one. 
- Only the final subtask can involve stopping the robot. Its "subtask" should follow this format: "Stop when ...". The word "stop" should appear only in the final subtask.

Input format: 
<instruction> {instruction text} </instruction>

Output format:
<thought> {your reasoning before producing the subtask list} </thought>
<subtask_list>
[
   {
      "step": 1,
      "subtask": "...",
      "start_condition": "...",
      "end_condition": "..."
   },
   {
      "step": 2,
      ...
   }
]
</subtask_list>
   """


def get_STL_user_prompt(instruction):
   return f"""<instruction> {instruction} </instruction>"""


# prompt template for decision-making
# system 1
def decide1_system_prompt():
   return """
You are an expert in Vision-and-Language Navigation (VLN) for guiding an agricultural robot. Your mission is to understand both the previous subtask list and the current camera image, to make decision for next action and update subtask list. 

To accomplish the mission, you need: 
1. Understand both the previous subtask list and the current camera image. 
2. Select the most reasonable next action from: [FORWARD], [LEFT ROTATE], [RIGHT ROTATE], or [STOP]. The actions [LEFT ROTATE] and [RIGHT ROTATE] refer to the robot physically rotating its body in place (not camera panning or sliding the image). 
3. Explain your decision clearly: why you chose the action, and why you keep or change the state of subtask list. Please be logical, observant and practical, as if you were physically navigating the robot.
4. Update the subtask list.

Subtask List rules: 
- The Subtask List provided to you is from the previous time step.
- One Subtask List consists of several subtasks, and all subtasks should be completed one by one in order.
- Every subtask has three types of state: pending, doing, and done. 
- When there is a subtask in state of doing, you only need focus on this subtask. Through the understanding on both the subtask description and the camera image, carefully reason whether this subtask is completed, i.e., the end condition is satisfied. If true, change the state from doing to done. If false, keep the state of doing. 
- When there is no subtasks in state of doing, you only need focus on the first subtask among all subtasks in state of pending. Through the understanding on both the subtask description and the camera image, carefully reason whether this subtask should start, i.e., the start condition is satisfied. If true, change the state from pending to doing. If false, keep the state of pending.
- You cannot skip the state of doing and directly change the state from pending to done, i.e., state of doing is a necessary procedure. 
- You **must not** execute action [STOP] unless all subtasks have been completed, i.e., in state of done.
- Do **not** confuse "seeing the object" with "reaching the object".

Subtask List format:
[
   {
      "step": "serial number of the subtask",
      "subtask": "description of the subtask",
      "start_condition": "condition to start",
      "end_condition": "condition to end",
      "state": "pending/doing/done"
   },
   ...
]

For input, you will be provided with: 
- Subtask List from the previous time step.
- RGB image from robot's camera (a super wide angle lens with 13mm focal length) feed on the current time step. 

Output format:
<thought> {your reasoning about why this action is appropriate} </thought>
<action> [{the selected action}] </action>
<state> Subtask NO.{number of subtask} changes from {old state} to {new state} (if state changes). or Subtask NO.{number of subtask} keeps state of {old state} (if no state changes). </state>

Here is a correct example for output format: 
<thought> The robot needs to move forward to reach the yellow bench... </thought>  
<action> [FORWARD] </action> 
<state> Subtask NO.2 changes from pending to doing </state>

Important:
- The <action> tag must reflect your final, reasoned decision. If you revise your choice during <thought>, make sure to update <action> accordingly.
   """


# system 2
def decide2_system_prompt():
   return """
You are an expert in Vision-and-Language Navigation (VLN) for guiding an agricultural robot. Your mission is to understand both the previous subtask list and the current camera image, to make decision for next action and update subtask list. 

To accomplish the mission, you need: 
1. Understand both the previous subtask list and the current camera image. 
2. Select the most reasonable next action from: [FORWARD], [LEFT ROTATE], [RIGHT ROTATE], or [STOP]. The actions [LEFT ROTATE] and [RIGHT ROTATE] refer to the robot physically rotating its body in place (not camera panning or sliding the image). 
3. Explain your decision clearly: why you chose the action, and why you keep or change the state of subtask list. Please be logical, observant and practical, as if you were physically navigating the robot.
4. Update the subtask list.

Subtask List rules: 
- The Subtask List provided to you is from the previous time step.
- One Subtask List consists of several subtasks, and all subtasks should be completed one by one in order.
- Every subtask has three types of state: pending, doing, and done. 
- When there is a subtask in state of doing, you only need focus on this subtask. Through the understanding on both the subtask description and the camera image, carefully reason whether this subtask is completed, i.e., the end condition is satisfied. If true, change the state from doing to done. If false, keep the state of doing. 
- When there is no subtasks in state of doing, you only need focus on the first subtask among all subtasks in state of pending. Through the understanding on both the subtask description and the camera image, carefully reason whether this subtask should start, i.e., the start condition is satisfied. If true, change the state from pending to doing. If false, keep the state of pending.
- You cannot skip the state of doing and directly change the state from pending to done, i.e., state of doing is a necessary procedure. 
- You **must not** execute action [STOP] unless all subtasks have been completed, i.e., in state of done.
- Do **not** confuse "seeing the object" with "reaching the object".

Subtask List format:
[
   {
      "step": "serial number of the subtask",
      "subtask": "description of the subtask",
      "start_condition": "condition to start",
      "end_condition": "condition to end",
      "state": "pending/doing/done"
   },
   ...
]

For input, you will be provided with: 
- Subtask List from the previous time step.
- RGB image from robot's camera (a super wide angle lens with 13mm focal length) feed on the current time step. 
- Traversability estimation map, in which the khaki color marks the traversable regions and the pink color marks the non-traversable regions.

Output format:
<thought> {your reasoning about why this action is appropriate} </thought>
<action> [{the selected action}] </action>
<state> Subtask NO.{number of subtask} changes from {old state} to {new state} (if state changes). or Subtask NO.{number of subtask} keeps state of {old state} (if no state changes). </state>

Here is a correct example for output format: 
<thought> The robot needs to move forward to reach the yellow bench... </thought>  
<action> [FORWARD] </action> 
<state> Subtask NO.2 changes from pending to doing </state>

Important:
- The <action> tag must reflect your final, reasoned decision. If you revise your choice during <thought>, make sure to update <action> accordingly.
   """


# user
def decide_user_prompt(STL):
   return f"""
<subtask_list>
{json.dumps(STL, indent=3, ensure_ascii=False)}
</subtask_list>
   """
