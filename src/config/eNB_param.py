from eNB import eNB

# enb_1 = eNB(1, (50, 150))
# enb_2 = eNB(2, (160, 330))
# enb_3 = eNB(3, (300, 220))
# enb_4 = eNB(4, (400, 400))
# enb_5 = eNB(5, (500, 190))
# enb_6 = eNB(6, (650, 300))
# # enb_7 = eNB(8, (780, 230))
# enb_7 = eNB(7, (900, 170))

# eNB_1_list  = [enb_1, enb_2, enb_3, enb_4, enb_5, enb_6, enb_7]

# enb_1 = eNB(1, (50, 150))
# enb_2 = eNB(2, (300, 330))
# enb_3 = eNB(3, (550, 190))
# enb_4 = eNB(4, (750, 300))
# enb_5 = eNB(5, (950, 170))

# eNB_1_list  = [enb_1, enb_2, enb_3, enb_4, enb_5]

# enb_1 = eNB(1, (50, 150))
# enb_2 = eNB(2, (160, 330))
# enb_3 = eNB(3, (300, 220w))
# enb_4 = eNB(4, (400, 400))
# enb_5 = eNB(5, (500, 190))
# enb_6 = eNB(6, (650, 300))
# # enb_7 = eNB(8, (780, 230))
# enb_7 = eNB(7, (900, 170))
# enb_8 = eNB(8, (1050, 330))
# enb_9 = eNB(9, (1200, 220))
# enb_10 = eNB(10, (1350, 270))
# enb_11 = eNB(11, (1500, 190))
# enb_12 = eNB(12, (1650, 280))
# enb_13 = eNB(13, (1800, 170))
# enb_14 = eNB(14, (1950, 400))
# enb_15 = eNB(15, (2290, 200))
# enb_16 = eNB(16, (2400, 350))
# enb_17 = eNB(17, (2550, 110))
# enb_18 = eNB(18, (2850, 300))

# eNB_1_list  = [enb_1, enb_2, enb_3, enb_4, enb_5, enb_6, enb_7, enb_8, enb_9, enb_10, enb_11, enb_12, enb_13, enb_14, enb_15, enb_16, enb_17, enb_18]

enb_1 = eNB(1, (50, 150))
enb_2 = eNB(2, (170, 330))  # Moved slightly to force earlier handover
enb_3 = eNB(3, (320, 210))  # Closer to road edge to trigger weak signal
enb_4 = eNB(4, (430, 400))  # Raised slightly for better overlap
enb_5 = eNB(5, (550, 180))  # Adjusted for weaker transition area
enb_6 = eNB(6, (700, 290))  # Closer to black spot area
enb_7 = eNB(7, (880, 160))  # Lower to create an artificial weak zone
enb_8 = eNB(8, (1040, 330))  
enb_9 = eNB(9, (1190, 210))  # Pushed lower for more dropouts  
enb_10 = eNB(10, (1330, 260))  # Shifted for weak transitions  
enb_11 = eNB(11, (1480, 190))  # Close to 1330 to induce failures  
enb_12 = eNB(12, (1630, 285))  # Adjusted near handover zone  
enb_13 = eNB(13, (1770, 155))  # Lowered to increase handover drop chances  
enb_14 = eNB(14, (1940, 400))  
enb_15 = eNB(15, (2230, 180))  # Slightly offset from road  
enb_16 = eNB(16, (2385, 340))  # Just outside road  
enb_17 = eNB(17, (2630, 115))  # Moved past road gap, will cause failures  
enb_18 = eNB(18, (2830, 300))  # Higher up to create weak transition  

eNB_1_list = [enb_1, enb_2, enb_3, enb_4, enb_5, enb_6, enb_7, enb_8, 
              enb_9, enb_10, enb_11, enb_12, enb_13, enb_14, enb_15, enb_16, 
              enb_17, enb_18]


enb_100 = eNB(1, (50, 150))
enb_101 = eNB(2, (190, 330))  # Moved slightly to force earlier handover
enb_102 = eNB(3, (550, 180))  # Adjusted for weaker transition area
enb_103 = eNB(4, (700, 290))  # Closer to black spot area
enb_104 = eNB(5, (880, 160))  # Lower to create an artificial weak zone
enb_105 = eNB(6, (1040, 330))
enb_106 = eNB(7, (1190, 210))  # Pushed lower for more dropouts
enb_107 = eNB(8, (1330, 260))  # Shifted for weak transitions\
enb_108 = eNB(9, (1630, 285))  # Adjusted near handover zone
enb_109 = eNB(10, (1770, 155))  # Lowered to increase handover drop chances 
enb_110 = eNB(11, (1940, 400))
enb_111 = eNB(12, (2230, 180))  # Slightly offset from road
enb_112 = eNB(13, (2385, 340))  # Just outside road
enb_113 = eNB(14, (2630, 115))  # Moved past road gap, will cause failures
enb_114 = eNB(15, (2830, 300))  # Higher up to create weak transition

eNB_2_list = [enb_100, enb_101, enb_102, enb_103, enb_104, enb_105, enb_106, enb_107,
              enb_108, enb_109, enb_110, enb_111, enb_112, enb_113, enb_114]