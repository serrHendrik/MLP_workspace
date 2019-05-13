Uses Python 3.7

Make sure to get the required packages with:
'pip install -r requirements.txt'

Use
'agent.cmd <port>'
to start an agent on the given port with the default settings.

For more options, agent_controller.py can be run directly. 
'python agent_controller.py <port> <extra arguments...>'
The possible arguments are:
*<port>: the port to run the agent on
*--verbose: verbose output
*--quiet: quiet output
*--type FR/IA: choose between Inequity Averse or Free Rider agent
*--train: Turn on training while playing
*--model <model>: Choose a model to use (models can be found in the models/ folder) 
