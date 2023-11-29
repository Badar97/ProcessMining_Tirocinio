# Execution Time: 0.001 s
v 1 START
v 2 RequestForPaymentSUBMITTEDbyEMPLOYEE
v 3 RequestForPaymentAPPROVEDbyADMINISTRATION
v 4 RequestForPaymentAPPROVEDbyBUDGETOWNER
v 5 RequestForPaymentFINALAPPROVEDbySUPERVISOR
v 6 RequestPayment
v 7 PaymentHandled
v 8 END

e 1 2 START__RequestForPaymentSUBMITTEDbyEMPLOYEE
e 2 3 RequestForPaymentSUBMITTEDbyEMPLOYEE__RequestForPaymentAPPROVEDbyADMINISTRATION
e 3 4 RequestForPaymentAPPROVEDbyADMINISTRATION__RequestForPaymentAPPROVEDbyBUDGETOWNER
e 4 5 RequestForPaymentAPPROVEDbyBUDGETOWNER__RequestForPaymentFINALAPPROVEDbySUPERVISOR
e 5 6 RequestForPaymentFINALAPPROVEDbySUPERVISOR__RequestPayment
e 5 7 RequestForPaymentFINALAPPROVEDbySUPERVISOR__PaymentHandled
e 6 8 RequestPayment__END
e 7 8 PaymentHandled__END
