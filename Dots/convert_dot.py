import pydot

for i in range(1,7):
	try:
		dot_file = 'gini_model_' + str(i) + '.dot'
		(graph,) = pydot.graph_from_dot_file(dot_file)
		png_file = 'gini_model_' + str(i) + '.png'
		graph.write_png(png_file)
	except:
		print("Cannot Find File" + dot_file)