# Execution Time: 0.005 s
v 1 START
v 2 RequestForPaymentSUBMITTEDbyEMPLOYEE
v 3 RequestForPaymentAPPROVEDbyADMINISTRATION
v 4 RequestForPaymentFINAL-APPROVEDbySUPERVISOR
v 5 RequestPayment
v 6 PaymentHandled
v 7 END

e 1 2 START__RequestForPaymentSUBMITTEDbyEMPLOYEE
e 2 3 RequestForPaymentSUBMITTEDbyEMPLOYEE__RequestForPaymentAPPROVEDbyADMINISTRATION
e 3 4 RequestForPaymentAPPROVEDbyADMINISTRATION__RequestForPaymentFINAL-APPROVEDbySUPERVISOR
e 4 5 RequestForPaymentFINAL-APPROVEDbySUPERVISOR__RequestPayment
e 5 6 RequestPayment__PaymentHandled
e 6 7 PaymentHandled__END
