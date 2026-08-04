'''
Normal global fitting, but with different ko_minus values below vs. above m_star.
'''
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits')
from richards import load_data
