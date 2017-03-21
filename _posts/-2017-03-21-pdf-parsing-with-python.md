---
layout: post
title: "PDF Parsing with Python"
date: 2017-03-21
header: true
footer: true
comments: true
tags: data scraping, text analysis, munging, textract, pdf mining
---

In this post, we'll parse my ConEdison Bill in Python using `textract`. The bill is three pages, each in a different format. Luckily, `textract` manages to get data seperated into different lines. We'll be working through a list of strings, where each string is typically one line in the pdf file.

To get figures from the document, the simplest solution would be to use `enumerate` to index every line of the scraped document, and then to use fixed references to the line where each amount appers. Unfortunately, this solution makes a drastic assumption that every bill will have the exact same number of lines in the exact same order with the exact same spacings.

Instead, we use a slightly harder but more robust solution. Our script assumes only that the distances between figures and their accompanying labels remain constant between bills. Our general technique is to scan the lines for a keyword, and then to grab the value that follows that keyword some number of lines following (usually two to four lines).



## Packages


```python
import os
import re
import argparse
import sys
from datetime import datetime
```

## User-modified variables


```python
filename = '9_19_2016.pdf'
```

# First page

<img src='/images/pdf-parsing-with-python/bill.png' style='width: 100%; object-fit: contain'/>

## Helper functions


```python
def get_kth_line_after_nth_occurrence(corpus, string, k=0, n=1):
    """Return kth line after the nth observation of string."""
    counter = 0
    for i, line in enumerate(corpus):
        if string in line:
            counter += 1
            if counter == n:
                return corpus[i + k]
```


```python
def get_rate_of_nth_occurrence(corpus, string, n=1):
    """Returns kwh rate of the line containing string.
    
    Works like get_kth_line_after_nth_occurrence() where k=0, but 
    we incorporate splitting the line at '@' which precedes the 
    rate, and at '\\', which follows the rate.
    
    We also trim any trailing '%' sign, so we can use it for sales tax.
    """
    counter = 0
    for line in corpus:
        if string in line:
            counter += 1
            if counter == n:
                _, rate = line.split('@')
                break

    rate = rate.split('\\')[0]
    
    # This is to remove the trailing '%' sign for sales tax rate
    while not rate[-1].isnumeric():
        rate = rate[:-1]
    
    rate = float(rate)
    return rate
```


```python
def strip_dollars(item):
    """Return float of string after stripping hyphens and dollars."""
    if isinstance(item, list):
        return [strip_dollars(i) for i in item]

    if item == 'None':
        return 0.0
    else:
        item = re.sub('\$', '', item)
        item = float(item)

    return item
```

## Get lines as a list of strings


```python
import textract
```


```python
text = textract.process(filename)
```


```python
text[:50]
```




    b'MATTHEW PETERSEN\n\nMessage Center\n\nYour account num'




```python
text = str(text)
```


```python
text[:50]
```




    "b'MATTHEW PETERSEN\\n\\nMessage Center\\n\\nYour accou"




```python
text = text[2:]  # remove b
```


```python
text[:50]
```




    'MATTHEW PETERSEN\\n\\nMessage Center\\n\\nYour account'




```python
lines = text.split('\\n')
```


```python
lines[:11]
```




    ['MATTHEW PETERSEN',
     '',
     'Message Center',
     '',
     'Your account number: 48-4305-1560-0006-7',
     '',
     'Counter',
     '',
     'Service delivered to: 560 W 163 St 35',
     'Your electric rate: EL1 Residential or Religious',
     'Your gas rate: GS1 Residential or Religious']



_Don't remove the empty strings, because they nicely divide some sections, that are difficult to separate otherwise._

# Client details

<img src='/images/pdf-parsing-with-python/client-details.png' style='height: 100%; object-fit: contain'/>

## Name


```python
full_name = lines[0]
first_name, last_name = full_name.split()
```


```python
first_name
```




    'MATTHEW'




```python
last_name
```




    'PETERSEN'




```python
full_name
```




    'MATTHEW PETERSEN'



## Account number


```python
account_number = get_kth_line_after_nth_occurrence(
    lines, 'Your account number:')
account_number = account_number.split(': ')[1]
account_number
```




    '48-4305-1560-0006-7'




```python
account_number_clean = re.sub('-', '', account_number)
account_number_clean
```




    '484305156000067'



## Address


```python
address = get_kth_line_after_nth_occurrence(
    lines, 'Service delivered to:')
address = address.split(': ')[1]
address
```




    '560 W 163 St 35'



## Electric rate


```python
electric_rate = get_kth_line_after_nth_occurrence(
    lines, 'Your electric rate:')
electric_rate = electric_rate.split(': ')[1]
electric_rate
```




    'EL1 Residential or Religious'



## Gas rate


```python
gas_rate = get_kth_line_after_nth_occurrence(
    lines, 'Your gas rate:')
gas_rate = gas_rate.split(': ')[1]
gas_rate
```




    'GS1 Residential or Religious'



## Next meter reading date


```python
next_meter_reading_date = get_kth_line_after_nth_occurrence(
    lines, 'Next meter reading date:')
next_meter_reading_date = next_meter_reading_date.split(': ')[1]
next_meter_reading_date
```




    'Tuesday, Oct 18, 2016'



# Billing summary section

A bit harder. Here's what it looks like

<img src='/images/pdf-parsing-with-python/billing-summary.png' style='height: 100%; object-fit: contain'/>

The issue is that, in the text parse, the numbers themselves are divorced from their descriptions. The lines of our parse go first down the left column of descriptions, and then down the right column of figures.

We _could_ use fixed indexing to match descriptions to figures, but this will fail if another bill has a different number of categories (such as a new one, 'past-due amount').

So we have to use dynamic indexing.

## Previous charges and payments


```python
# Find start of 'previous charges' section
for i, line in enumerate(lines):
    if 'Your previous charges and payments' in line:
        previous_charges_start = i
        break

# Find start of descriptions (assumes start immediately)
descriptions_start = previous_charges_start + 1

# Find start of amounts (assumes start after next empty line)
for i, line in enumerate(lines[descriptions_start:]):
    if not line:
        descriptions_end = descriptions_start + i
        amounts_start = descriptions_end + 1
        break

# Find end of amounts (assumes end at next empty line)
for i, line in enumerate(lines[amounts_start:]):
    if not line:
        amounts_end = amounts_start + i
        break
```


```python
descriptions = lines[descriptions_start:descriptions_end]
descriptions
```




    ['Total charges from your last bill',
     'Payments through Sep 15, than you',
     'Remaining balance']




```python
amounts = lines[amounts_start:amounts_end]
amounts = strip_dollars(amounts)
amounts
```




    [82.18, -82.18, 0.0]




```python
previous_charges_and_payments = {
    desc: amt for desc, amt in zip(descriptions, amounts)
}
previous_charges_and_payments
```




    {'Payments through Sep 15, than you': -82.18,
     'Remaining balance': 0.0,
     'Total charges from your last bill': 82.18}



## New charges


```python
# Find new charges section
for i, line in enumerate(lines):
    if 'Your new charges' in line:
        new_charges_start = i
        break

# Find start of descriptions in new charges section
for i, line in enumerate(lines[new_charges_start:]):
    if not line:
        descriptions_start = new_charges_start + (i + 1)
        break

# Find start of amounts in new charges section
for i, line in enumerate(lines[descriptions_start:]):
    if not line:
        descriptions_end = descriptions_start + i
        amounts_start = descriptions_end + 1
        break

# Find end of amounts in new charges section
for i, line in enumerate(lines[amounts_start:]):
    if not line:
        amounts_end = amounts_start + i
        break
```


```python
descriptions = lines[descriptions_start:descriptions_end]
descriptions
```




    ['Electricity charges - for 30 days', 'Gas charges - for 30 days']




```python
amounts = lines[amounts_start:amounts_end]
amounts = strip_dollars(amounts)
amounts
```




    [67.72, 23.67]




```python
new_charges = {desc: amt for desc, amt in zip(descriptions, amounts)}
new_charges
```




    {'Electricity charges - for 30 days': 67.72,
     'Gas charges - for 30 days': 23.67}



## Total new charges


```python
total_new_charges = get_kth_line_after_nth_occurrence(
    lines, 'Total new charges', k=2)
total_new_charges = strip_dollars(total_new_charges)
total_new_charges
```




    91.39




```python
# Add it to our dictionary
new_charges['Total new charges'] = total_new_charges
new_charges
```




    {'Electricity charges - for 30 days': 67.72,
     'Gas charges - for 30 days': 23.67,
     'Total new charges': 91.39}




```python
# Double check that it equals the sum of other charges
assert sum(new_charges.values()) == 2 * new_charges['Total new charges'], \
    "Total new charges does not equal sum of listed new charges"
```

## Billing period


```python
billing_period = get_kth_line_after_nth_occurrence(lines, 'Billing period')
billing_period = billing_period.split(': ')[1]
billing_period = billing_period.split(' to ')
billing_period
```




    ['Aug 17, 2016', 'Sep 16, 2016']




```python
# Cast billing period to datetime objects
fmt = '%b %d, %Y'

start = datetime.strptime(billing_period[0], fmt)
end = datetime.strptime(billing_period[1], fmt)

billing_period = start, end
billing_period
```




    (datetime.datetime(2016, 8, 17, 0, 0), datetime.datetime(2016, 9, 16, 0, 0))



## Days in billing period


```python
days_in_billing_period = (end - start).days
days_in_billing_period
```




    30



## Meter number


```python
meter_number = get_kth_line_after_nth_occurrence(lines, 'Meter#')
meter_number = meter_number.split()[1]
meter_number
```




    '7543491'



## kWh used


```python
kwh_used = get_kth_line_after_nth_occurrence(
    lines, 'Electricity you used', k=4)
kwh_used = kwh_used.split()[0]
kwh_used = float(kwh_used)
kwh_used
```




    236.0



---
# Supply charges (electricity)

## kWh rate (supply)


```python
cents_per_kwh_supply = get_rate_of_nth_occurrence(lines, 'Supply')
cents_per_kwh_supply
```




    7.5932



## Supply charge


```python
kwh_used
```




    236.0




```python
cents_per_kwh_supply
```




    7.5932




```python
# Divide by 100 to get dollars again
supply_charge_elec = (kwh_used * cents_per_kwh_supply) / 100
supply_charge_elec
```




    17.919952000000002



## Merchant function charge


```python
merchant_function_charge_elec = get_kth_line_after_nth_occurrence(
    lines, 'Your electricity use', 5, 1
)
merchant_function_charge_elec = \
    strip_dollars(merchant_function_charge_elec)
merchant_function_charge_elec
```




    1.19



## GRT and other tax surcharges (supply)


```python
# Both delivery and supply grt have the same description, so
# it's impossible to search for one or the other uniquely. We
# have to get them both together (delivery appears first in the
# text parse, although it's actually further in the document)

for i, line in enumerate(lines):
    if 'GRT & other tax surcharges' in line:
        grt_delivery_elec = lines[i + 2]
        continue_index = i + 1
        break    

for i, line in enumerate(lines[continue_index:]):
    if 'GRT & other tax surcharges' in line:
        grt_supply_elec = lines[continue_index + i + 2]
        index = i
        break
```


```python
grt_delivery_elec, grt_supply_elec = \
    strip_dollars([grt_delivery_elec, grt_supply_elec])
```


```python
grt_delivery_elec
```




    2.2




```python
grt_supply_elec
```




    0.46



## Total supply charges


```python
total_supply_charges_elec = get_kth_line_after_nth_occurrence(
    lines, 'Total supply charges', 2, 1
)
total_supply_charges_elec = strip_dollars(total_supply_charges_elec)
total_supply_charges_elec
```




    19.57




```python
# Double-check that it matches the sum of component charges

total_supply_charges_elec_cpt \
= supply_charge_elec \
+ merchant_function_charge_elec \
+ grt_supply_elec

total_supply_charges_elec_cpt = \
    round(total_supply_charges_elec_cpt, 2)

assert total_supply_charges_elec == \
    total_supply_charges_elec_cpt, \
    "Total supply charge does not equal sum of component charges"
```

---
# Delivery charges (electricity)

## Basic service charge


```python
basic_service_charge_elec = get_kth_line_after_nth_occurrence(
    lines, 'Basic service charge', 2, 1
)
basic_service_charge_elec = strip_dollars(basic_service_charge_elec)
basic_service_charge_elec
```




    16.36



## kWh rate (delivery)


```python
cents_per_kwh_delivery = get_rate_of_nth_occurrence(lines, 'Delivery')
cents_per_kwh_delivery
```




    10.5551



## Delivery charge


```python
delivery_charge_elec = (kwh_used * cents_per_kwh_delivery) / 100
delivery_charge_elec
```




    24.910036



## kWh rate (system benefit charge)


```python
cents_per_kwh_sys_ben = get_rate_of_nth_occurrence(lines, 'System Benefit Charge', n=1)
cents_per_kwh_sys_ben
```




    0.6186



## System benefit charge


```python
system_benefit_charge_elec = (kwh_used * cents_per_kwh_sys_ben) / 100
system_benefit_charge_elec
```




    1.4598959999999999



## kWh rate (Temporary NY State Surcharge)


```python
cents_per_kwh_temp_ny_st_sur = get_rate_of_nth_occurrence(lines, 'Temporary NY State', n=1)
cents_per_kwh_temp_ny_st_sur
```




    0.1271



## Temporary NY State Surcharge


```python
temporary_ny_state_surcharge_elec = \
    (kwh_used * cents_per_kwh_temp_ny_st_sur) / 100
temporary_ny_state_surcharge_elec
```




    0.29995599999999994



## GRT  & other tax surcharges (delivery)


```python
# From earlier, we had to get both at the same time
grt_delivery_elec
```




    2.2



## Total delivery charges


```python
total_delivery_charges_elec = get_kth_line_after_nth_occurrence(
    lines, 'Total delivery charges', k=2, n=1
)
total_delivery_charges_elec = strip_dollars(total_delivery_charges_elec)
total_delivery_charges_elec
```




    45.23




```python
# Check that total delivery charges equals the sum of components

total_delivery_charges_elec_cpt \
= basic_service_charge_elec \
+ delivery_charge_elec \
+ system_benefit_charge_elec \
+ temporary_ny_state_surcharge_elec \
+ grt_delivery_elec

total_delivery_charges_elec_cpt = \
    round(total_delivery_charges_elec_cpt, 2)

assert total_delivery_charges_elec \
    == total_delivery_charges_elec_cpt, \
    "Total delivery charges does not equal sum of components"
```

---
# Finishing electricity

## Sales tax rate


```python
sales_tax_rate_elec = get_rate_of_nth_occurrence(lines, 'Sales tax', n=1)
sales_tax_rate_elec /= 100
sales_tax_rate_elec
```




    0.045



## Sales tax


```python
sales_tax_elec = sales_tax_rate_elec \
    * (total_supply_charges_elec + total_delivery_charges_elec)
sales_tax_elec = round(sales_tax_elec, 2)
sales_tax_elec
```




    2.92



## Total electricity charges


```python
total_charges_elec = get_kth_line_after_nth_occurrence(
    lines, 'Total electricity charges', 2, 1
)
total_charges_elec = strip_dollars(total_charges_elec)
total_charges_elec
```




    67.72




```python
# Double check our figures

total_charges_elec_cpt \
= total_supply_charges_elec \
+ total_delivery_charges_elec \
+ sales_tax_elec

assert total_charges_elec_cpt == total_charges_elec, \
    "Total electricity charges does not equal sum of components"
```

---
# Gas

# Supply charges (gas)

## Therms used


```python
# WARNING: This is a hacky way to get therms used...
therms_used = get_kth_line_after_nth_occurrence(
    lines, 'Your average daily gas use', 2, 1
)
therms_used = therms_used.split()[0]
therms_used = float(therms_used)
therms_used
```




    4.0



## Cents per therm (supply)


```python
cents_per_therm_supply = get_rate_of_nth_occurrence(lines, 'therms @')
cents_per_therm_supply
```




    26.5



## Supply charge


```python
# Divide by 100 to get dollars
supply_charge_gas = (cents_per_therm_supply * therms_used) / 100
supply_charge_gas
```




    1.06



## Merchant function charge


```python
merchant_function_charge_gas = get_kth_line_after_nth_occurrence(
    lines, 'Merchant function charge', 2, 2
)
merchant_function_charge_gas = strip_dollars(merchant_function_charge_gas)
merchant_function_charge_gas
```




    0.07



## GRT & other tax surcharges


```python
grt_supply_gas = get_kth_line_after_nth_occurrence(
    lines, 'GRT & other tax surcharges', k=2, n=4
)
grt_supply_gas = strip_dollars(grt_supply_gas)
grt_supply_gas
```




    0.03



## Total supply charges (gas)


```python
total_supply_charges_gas = get_kth_line_after_nth_occurrence(
    lines, 'Total supply charges', k=2, n=2
)
total_supply_charges_gas = strip_dollars(total_supply_charges_gas)
total_supply_charges_gas
```




    1.16




```python
# Double check our work on supply charges for gas

total_supply_charges_gas_cpt \
= supply_charge_gas \
+ merchant_function_charge_gas \
+ grt_supply_gas

total_supply_charges_gas_cpt = round(total_supply_charges_gas_cpt, 2)

assert total_supply_charges_gas_cpt == total_supply_charges_gas, \
    "Total supply charges gas does not equal sum of components"
```

# Delivery charges (gas)

## Basic service (covers 3 therms)


```python
basic_service_charge_gas = get_kth_line_after_nth_occurrence(
    lines, 'Basic service charge', k=2, n=2
)
basic_service_charge_gas = strip_dollars(basic_service_charge_gas)
basic_service_charge_gas
```




    19.2



## Excess service charge (on excess of 3 therms)


```python
excess_therms = therms_used - 3.0

cents_per_therm_excess = get_rate_of_nth_occurrence(
    lines, 'Remaining %.1f therms' % excess_therms, n=1
)
cents_per_therm_excess
```




    97.0




```python
excess_service_charge_gas = \
    (excess_therms * cents_per_therm_excess) / 100
excess_service_charge_gas
```




    0.97



## Monthly rate adjustment (gas delivery)


```python
cents_per_therm_mthly_adj = get_rate_of_nth_occurrence(
    lines, 'Monthly rate adjustment @', n=1
)
cents_per_therm_mthly_adj
```




    3.5




```python
monthly_rate_adjustment_gas = (therms_used * cents_per_therm_mthly_adj) / 100
monthly_rate_adjustment_gas
```




    0.14



## System benefit charge (gas delivery)


```python
cents_per_therm_sys_ben_chg = get_rate_of_nth_occurrence(
    lines, 'System Benefit Charge @', n=2
)
cents_per_therm_sys_ben_chg
```




    1.25




```python
system_benefit_charge_gas = \
    (therms_used * cents_per_therm_sys_ben_chg) / 100
system_benefit_charge_gas
```




    0.05



## Temporary NY state surcharge (gas delivery)


```python
cents_per_therm_temp_ny_st_sur = get_rate_of_nth_occurrence(
    lines, 'Temporary NY State Surcharge @', n=2
)
cents_per_therm_temp_ny_st_sur
```




    2.0




```python
temporary_ny_state_surcharge_gas = \
    (therms_used * cents_per_therm_temp_ny_st_sur) / 100
temporary_ny_state_surcharge_gas
```




    0.08



## GRT & other tax surcharges (gas delivery)


```python
grt_delivery_gas = get_kth_line_after_nth_occurrence(
    lines, 'GRT & other tax surcharges', k=2, n=3
)
grt_delivery_gas = strip_dollars(grt_delivery_gas)
grt_delivery_gas
```




    1.05



## Total delivery charges (gas)


```python
# WARNING: it's strange for the value to follow 8 lines later
total_delivery_charges_gas = get_kth_line_after_nth_occurrence(
    lines, 'Total delivery charges', k=8, n=2
)
total_delivery_charges_gas = strip_dollars(total_delivery_charges_gas)
total_delivery_charges_gas
```




    21.49




```python
# Double check our work (gas delivery)

total_delivery_charges_gas_cpt \
= basic_service_charge_gas \
+ excess_service_charge_gas \
+ monthly_rate_adjustment_gas \
+ system_benefit_charge_gas \
+ temporary_ny_state_surcharge_gas \
+ grt_delivery_gas

total_delivery_charges_gas_cpt = round(total_delivery_charges_gas_cpt, 2)

assert total_delivery_charges_gas_cpt == total_delivery_charges_gas, \
    "Total delivery charges gas is not equal to the sum of its components"
```

# Finishing up gas

## Sales tax


```python
sales_tax_rate_gas = sales_tax_rate_elec

sales_tax_gas = sales_tax_rate_gas \
    * (total_supply_charges_gas + total_delivery_charges_gas)
sales_tax_gas = round(sales_tax_gas, 2)
sales_tax_gas
```




    1.02



## Total gas charges


```python
total_charges_gas = get_kth_line_after_nth_occurrence(
    lines, 'Total gas charges', k=2, n=1
)
total_charges_gas = strip_dollars(total_charges_gas)
total_charges_gas
```




    23.67




```python
# Double check our work

total_charges_gas_cpt \
= total_supply_charges_gas \
+ total_delivery_charges_gas \
+ sales_tax_gas

total_charges_gas_cpt = round(total_charges_gas_cpt, 2)

assert total_charges_gas_cpt == total_charges_gas, \
    "Total gas charges is not equal to sum of components"
```

# Write to CSV file


```python
import csv
```

### Helper function to make sure we only write each bill once


```python
def csv_column_has_id(csvfile, column_name, ID, sep=','):
    """Checks for presence of id in column_name of csvfile."""
    with open(csvfile, 'r') as csvfile:
        for i, line in enumerate(csvfile):
            if i == 0: # Header row: find number of ID column
                words = line.split(sep)
                id_column_number = words.index(column_name)
            else:
                words = line.split(sep)
                if words[id_column_number] == ID:
                    return True
    return False
```

## Variables to write


```python
# Unique ID of this dataset is the filename of the pdf.
# We will not write a new row to conedison.csv if this ID 
# exists already in the ID column
ID = filename.split('.')[0]

client = [
    ('first_name', first_name),
    ('last_name', last_name),
    ('full_name', full_name),
    ('account_number', account_number),
    ('account_number_clean', account_number_clean),
    ('address', address),
    ('electric_rate', electric_rate),
    ('gas_rate', gas_rate),
    ('meter_number', meter_number),
]

elec = [
    ('kwh_used', kwh_used),
    ('cents_per_kwh_supply', cents_per_kwh_supply),
    ('supply_charge_elec', supply_charge_elec),
    ('merchant_function_charge_elec', merchant_function_charge_elec),
    ('grt_supply_elec', grt_supply_elec),
    ('total_supply_charges_elec', total_supply_charges_elec),
    ('basic_service_charge_elec', basic_service_charge_elec),
    ('cents_per_kwh_delivery', cents_per_kwh_delivery),
    ('delivery_charge_elec', delivery_charge_elec),
    ('cents_per_kwh_sys_ben', cents_per_kwh_sys_ben),
    ('system_benefit_charge_elec', system_benefit_charge_elec),
    ('cents_per_kwh_temp_ny_st_sur', cents_per_kwh_temp_ny_st_sur),
    ('temporary_ny_state_surcharge_elec', temporary_ny_state_surcharge_elec),
    ('grt_delivery_elec', grt_delivery_elec),
    ('total_delivery_charges_elec', total_delivery_charges_elec),
    ('sales_tax_rate_elec', sales_tax_rate_elec),
    ('sales_tax_elec', sales_tax_elec),
    ('total_charges_elec', total_charges_elec),
]

gas = [
    ('therms_used', therms_used),
    ('cents_per_therm_supply', cents_per_therm_supply),
    ('supply_charge_gas', supply_charge_gas),
    ('merchant_function_charge_gas', merchant_function_charge_gas),
    ('grt_supply_gas', grt_supply_gas),
    ('total_supply_charges_gas', total_supply_charges_gas),
    ('basic_service_charge_gas', basic_service_charge_gas),
    ('cents_per_therm_excess', cents_per_therm_excess),
    ('excess_service_charge_gas', excess_service_charge_gas),
    ('cents_per_therm_mthly_adj', cents_per_therm_mthly_adj),
    ('monthly_rate_adjustment_gas', monthly_rate_adjustment_gas),
    ('cents_per_therm_sys_ben_chg', cents_per_therm_sys_ben_chg),
    ('system_benefit_charge_gas', system_benefit_charge_gas),
    ('cents_per_therm_temp_ny_st_sur', cents_per_therm_temp_ny_st_sur),
    ('temporary_ny_state_surcharge_gas', temporary_ny_state_surcharge_gas),
    ('grt_delivery_gas', grt_delivery_gas),
    ('total_delivery_charges_gas', total_delivery_charges_gas),
    ('sales_tax_rate_gas', sales_tax_rate_gas),
    ('sales_tax_gas', sales_tax_gas),
    ('total_charges_gas', total_charges_gas),
]
```

## Write data


```python
# If the file doesn't exist, make it...
if not os.path.isfile('conedison.csv'):
    os.mknod('conedison.csv')
    
    # ...then make a heading row of variable names
    with open('conedison.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['ID']
            + [tup[0] for tup in client]
            + [tup[0] for tup in elec]
            + [tup[0] for tup in gas]
        )
```


```python
# If our bill is not yet stored in the file
if not csv_column_has_id('conedison.csv', 'ID', ID):
    with open('conedison.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [str(ID)]
            + [str(tup[1]) for tup in client]
            + [str(tup[1]) for tup in elec]
            + [str(tup[1]) for tup in gas]
        )
```

# Try reading in our scraped data with Pandas


```python
import pandas as pd
```


```python
data = pd.read_csv('conedison.csv')
```


```python
data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>first_name</th>
      <th>last_name</th>
      <th>full_name</th>
      <th>account_number</th>
      <th>account_number_clean</th>
      <th>address</th>
      <th>electric_rate</th>
      <th>gas_rate</th>
      <th>meter_number</th>
      <th>...</th>
      <th>monthly_rate_adjustment_gas</th>
      <th>cents_per_therm_sys_ben_chg</th>
      <th>system_benefit_charge_gas</th>
      <th>cents_per_therm_temp_ny_st_sur</th>
      <th>temporary_ny_state_surcharge_gas</th>
      <th>grt_delivery_gas</th>
      <th>total_delivery_charges_gas</th>
      <th>sales_tax_rate_gas</th>
      <th>sales_tax_gas</th>
      <th>total_charges_gas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9_19_2016</td>
      <td>MATTHEW</td>
      <td>PETERSEN</td>
      <td>MATTHEW PETERSEN</td>
      <td>48-4305-1560-0006-7</td>
      <td>484305156000067</td>
      <td>560 W 163 St 35</td>
      <td>EL1 Residential or Religious</td>
      <td>GS1 Residential or Religious</td>
      <td>7543491</td>
      <td>...</td>
      <td>0.14</td>
      <td>1.25</td>
      <td>0.05</td>
      <td>2.0</td>
      <td>0.08</td>
      <td>1.05</td>
      <td>21.49</td>
      <td>0.045</td>
      <td>1.02</td>
      <td>23.67</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 48 columns</p>
</div>


# Summary

### Notes:

* Unless otherwise specified, all numbers are in dollars
* If document format changes, some numbers in this program must be changed (the parameters input to helper functions through the document), but nothing else. There's inherent mess in text-parsing a PDF file, and counting the lines between figures and their matching descriptions was the simple solution.
* If ConEdison does not actually round intermediate calculations, then assertion checks in this program may fail by being off by one cent. This shouldn't matter, because we always use the total printed on ConEdison's bill, and are only assertion checking to make sure our intermediate figures are correct, should we want to use them later.



```python

```
