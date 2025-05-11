\* FuelOptimization_MyFreightliner_Fresno_to_Cheshire_Excel *\
Minimize
OBJ: 4.2479 FuelPurchased_1 + 4.1318 FuelPurchased_2 + 4.2059 FuelPurchased_3
 + 4.1771 FuelPurchased_4
Subject To
DesiredEndFuelConstraint: FuelAtArrival_5 >= 75
FuelBalanceAfterPurchase_374: FuelAfterPurchase_4 - FuelAtArrival_4
 - FuelPurchased_4 = 0
FuelBalanceAfterPurchase_382: FuelAfterPurchase_1 - FuelAtArrival_1
 - FuelPurchased_1 = 0
FuelBalanceAfterPurchase_392: FuelAfterPurchase_2 - FuelAtArrival_2
 - FuelPurchased_2 = 0
FuelBalanceAfterPurchase_755: FuelAfterPurchase_3 - FuelAtArrival_3
 - FuelPurchased_3 = 0
FuelFlow_374_to_end: - FuelAfterPurchase_4 + FuelAtArrival_5 = -92.18460454
FuelFlow_382_to_392: - FuelAfterPurchase_1 + FuelAtArrival_2 = -58.42789337
FuelFlow_392_to_755: - FuelAfterPurchase_2 + FuelAtArrival_3 = -36.26889091
FuelFlow_755_to_374: - FuelAfterPurchase_3 + FuelAtArrival_4 = -11.45125877
FuelFlow_start_to_382: - FuelAfterPurchase_0 + FuelAtArrival_1 = -26.97495241
InitialFuelConstraint: FuelAtArrival_0 = 150
LinkPurchaseToStopDecisionLower_374: FuelPurchased_4 - 0.01 StopDecision_4
 >= 0
LinkPurchaseToStopDecisionLower_382: FuelPurchased_1 - 0.01 StopDecision_1
 >= 0
LinkPurchaseToStopDecisionLower_392: FuelPurchased_2 - 0.01 StopDecision_2
 >= 0
LinkPurchaseToStopDecisionLower_755: FuelPurchased_3 - 0.01 StopDecision_3
 >= 0
LinkPurchaseToStopDecisionUpper_374: FuelPurchased_4 - 750 StopDecision_4 <= 0
LinkPurchaseToStopDecisionUpper_382: FuelPurchased_1 - 750 StopDecision_1 <= 0
LinkPurchaseToStopDecisionUpper_392: FuelPurchased_2 - 750 StopDecision_2 <= 0
LinkPurchaseToStopDecisionUpper_755: FuelPurchased_3 - 750 StopDecision_3 <= 0
MaxStopsConstraint: StopDecision_1 + StopDecision_2 + StopDecision_3
 + StopDecision_4 <= 0
NoPurchaseAtStartConstraint: FuelAfterPurchase_0 - FuelAtArrival_0 = 0
NoPurchaseNonStation_end: FuelAfterPurchase_5 - FuelAtArrival_5 = 0
NoPurchaseNonStation_start: FuelAfterPurchase_0 - FuelAtArrival_0 = 0
PurchaseCapacityLimit_374: FuelAtArrival_4 + FuelPurchased_4 <= 750
PurchaseCapacityLimit_382: FuelAtArrival_1 + FuelPurchased_1 <= 750
PurchaseCapacityLimit_392: FuelAtArrival_2 + FuelPurchased_2 <= 750
PurchaseCapacityLimit_755: FuelAtArrival_3 + FuelPurchased_3 <= 750
SufficientFuelForSegment_374_to_end: FuelAfterPurchase_4 >= 142.18460454
SufficientFuelForSegment_382_to_392: FuelAfterPurchase_1 >= 108.42789337
SufficientFuelForSegment_392_to_755: FuelAfterPurchase_2 >= 86.26889091
SufficientFuelForSegment_755_to_374: FuelAfterPurchase_3 >= 61.45125877
SufficientFuelForSegment_start_to_382: FuelAfterPurchase_0 >= 76.97495241
ZeroPurchaseAtStartPOI: FuelPurchased_0 = 0
ZeroPurchaseNonStation_end: FuelPurchased_5 = 0
ZeroPurchaseNonStation_start: FuelPurchased_0 = 0
Bounds
 50 <= FuelAfterPurchase_0 <= 750
 50 <= FuelAfterPurchase_1 <= 750
 50 <= FuelAfterPurchase_2 <= 750
 50 <= FuelAfterPurchase_3 <= 750
 50 <= FuelAfterPurchase_4 <= 750
 50 <= FuelAfterPurchase_5 <= 750
 50 <= FuelAtArrival_0 <= 750
 50 <= FuelAtArrival_1 <= 750
 50 <= FuelAtArrival_2 <= 750
 50 <= FuelAtArrival_3 <= 750
 50 <= FuelAtArrival_4 <= 750
 50 <= FuelAtArrival_5 <= 750
Binaries
StopDecision_1
StopDecision_2
StopDecision_3
StopDecision_4
End
