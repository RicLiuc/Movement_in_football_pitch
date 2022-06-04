

def get_data(team):

	name = "Files/"+str(team)+"/"+str(team)+"Data.h5"

	return name


def get_dataset(team):

	name = "Files/"+str(team)+"/"+str(team)+"Dataset.h5"

	return name


def prediction(team):

	name = "Files/"+str(team)+"/"+str(team)+"Prediction.h5"

	return name


def get_ID(team):

	name = "Files/"+str(team)+"/"+str(team)+"ID.h5"

	return name


def get_MinMax(team):

	name = "Files/"+str(team)+"/"+str(team)+"mm.txt"

	return name


def get_model(team, nb_residual_unit, len_closeness, len_period):

	name = "Models/"+str(team)+"/L"+str(nb_residual_unit)+"_C"+str(len_closeness)+"_P"+str(len_period)+"/best.pt" 

	return name


def get_model_path(team, nb_residual_unit, len_closeness, len_period):

	name = "Models/"+str(team)+"/L"+str(nb_residual_unit)+"_C"+str(len_closeness)+"_P"+str(len_period)

	return name


def get_match_info(match_id):

	name = "Matches/Data/"+str(match_id)+"Data_match.json"

	return name

def get_external(team):

	name = "Files/"+str(team)+"/"+str(team)+"External.h5"

	return name



