input_list = [[100, 200], [200, 300], [300, 400]]
target_list = [[300], [400], [500]]

input_template = ""
for i, input in enumerate(input_list):
    input_template += "thick_schaefer_" + str(input) + "_7 "
    input_template += "vol_schaefer_" + str(input) + "_7"
