from example_step_definition import ws


# Once profiling is done, we can train the model we passed to the workflow step, it will save the model into a file
ws.train()

# After profiling, we can print the objective function from the performance model
print(ws.get_objective_function())
