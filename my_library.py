def test_load():
  return 'loaded'

def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, evidence, evidence_val, target, target_val):
  t_subset = up_table_subset(table, target, 'equals', target_val)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_val else 0 for v in e_list])/len(e_list)
  return p_b_a + .01  #completely incorrect Laplace smoothing factor

def cond_probs_product(table, evidence_vals, target_col, target_val):
  evidence_cols = up_list_column_names(table)[:-1]
  evidence_complete = up_zip_lists(evidence_cols, evidence_vals)
  cond_prob_list = [cond_prob(table,column,val,target_col,target_val) for column,val in evidence_complete]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator

def prior_prob(table, col, val):
  t_list = up_get_column(table, col)
  prob = sum([1 if v==val else 0 for v in t_list])/len(t_list)
  return prob

def naive_bayes(table, evidence_row, target):
  #compute P(Flu=0|...) by collecting cond_probs in a list, take the produce of the list, finally multiply by P(Flu=0)
  p_zero = cond_probs_product(table, evidence_row, target, 0) * prior_prob(table, target, 0)

  #do same for P(Flu=1|...)
  p_one = cond_probs_product(table, evidence_row, target, 1) * prior_prob(table, target, 1)

  #Use compute_probs to get 2 probabilities
  neg, pos = compute_probs(p_zero,p_one)
  
  #return your 2 results in a list
  return [neg, pos]

def metrics(predictionPairsList):
  #check formatting of pairs list
  assert isinstance(predictionPairsList,list), 'prediction pairs list mut be a list'
  for item in predictionPairsList:
    assert isinstance(item,list), 'prediction pairs must each be lists'
    assert len(item) == 2, 'prediction pairs must have exactly two items'
    assert isinstance(item[0],int), 'prediction pairs must contain only integers'
    assert isinstance(item[1],int), 'prediction pairs must contain only integers'
    assert item[0] >= 0, 'prediction pair values must be non-negative'
    assert item[1] >= 0, 'prediction pair values must be non-negative'

  
  #calculate core of matrix
  tn = sum([1 if pair==[0,0] else 0 for pair in predictionPairsList])
  tp = sum([1 if pair==[1,1] else 0 for pair in predictionPairsList])
  fp = sum([1 if pair==[1,0] else 0 for pair in predictionPairsList])
  fn = sum([1 if pair==[0,1] else 0 for pair in predictionPairsList])

  #calculate accuraccy
  if len(predictionPairsList) == 0:
    print("warning: empty list!");
    accuracy = 0
  else:
    accuracy = sum([1 if x==y else 0 for x,y in predictionPairsList])/len(predictionPairsList)
  
  #calculate precision and recall
  if tp + fp == 0:
    precision = 0
  else:
    precision = tp / (tp + fp)
  if tp + fn == 0:
    recall = 0
  else:
    recall = tp / (tp + fn)

  #calculate f1 value
  if precision + recall == 0:
    f1 = 0
  else:
    f1 = (2 * precision * recall) / (precision + recall)

  #finally, collect values in a dictionary and return it
  dictionary = {'Accuracy': accuracy, 'F1': f1, 'Precision': precision, 'Recall': recall}
  return dictionary
