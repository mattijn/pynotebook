// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// Execute NDAI VHI Query using the coordinates and parameter setting through PyWPS

function execute_ndai_vhi_Query(lonlat) {

        document.getElementById("ts_nvai_vci_options").style.display = 'none';
        document.getElementById("ts_ntai_tci_options").style.display = 'none';
        document.getElementById("ts_ndai_vhi_options").style.display = 'block';
        document.getElementById("ts_vegetation_options").style.display = 'none';
        document.getElementById("ts_lst_options").style.display = 'none';

        document.getElementById("ts_nvai_vci_refresh").style.display = 'none';
        document.getElementById("ts_ntai_tci_refresh").style.display = 'none';
        document.getElementById("ts_ndai_vhi_refresh").style.display = 'block';
        document.getElementById("ts_vegetation_refresh").style.display = 'none';
        document.getElementById("ts_lst_refresh").style.display = 'none';
        document.getElementById("ts_precipitation_refresh").style.display = 'none';

    if ($('#startDate_ts').length > 0) {
        // dostuff
        ts_start_date = $('#startDate_ts').MonthPicker('GetSelectedYear') + '-' + pad($(
            '#startDate_ts').MonthPicker('GetSelectedMonth')) + '-' + '01';
    } else {
        ts_start_date = '2010-01-01';
        console.log('date_in not undefined, set to default:');
    };
	
    if ($('#endDate_ts').length > 0) {
        // dostuff
        ts_end_date = $('#endDate_ts').MonthPicker('GetSelectedYear') + '-' + pad($(
            '#endDate_ts').MonthPicker('GetSelectedMonth')) + '-' + '01';

    } else {
        ts_end_date = '2012-01-01';
        console.log('date_in not undefined, set to default:');
    };	


// check if start_date is after the end_date
if (ts_start_date > ts_end_date) {
    // your code here.
    var $message_startDate_after_endDate = $('<div></div>')

        .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> Function 2/3 (Using NDVI&LST): The start date is after the end date!')
        .dialog({
        dialogClass: "no-close",
        modal: true,
        resizable: false,

        title: 'Warning',
        buttons: {
            OK: function () {
                $(this).dialog("close");
                $message_startDate_after_endDate.remove();
                $('#ts_ndai_vhi_refresh').attr('src', './js/img/refresh.png');
            }
        }
    });


    $message_startDate_after_endDate.dialog('open').parent().addClass("ui-state-error");

    return
};

// check if start_date is after the available data according to metadata
// check if dataset is available based on metadata
if (typeof metadata_NDVI_MOD13C1005_to_date !== 'undefined') {    
    if (ts_start_date > metadata_NDVI_MOD13C1005_to_date) {
        // your code here.

        var $message_metadata_NDVI_MOD13C1005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> Function 2/3 (NDVI): The start date is after the available date! Dataset availability: ' + metadata_NDVI_MOD13C1005_from_date + ' to ' + metadata_NDVI_MOD13C1005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_NDVI_MOD13C1005.remove();
                    $('#ts_ndai_vhi_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_NDVI_MOD13C1005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    run_the_ndai_vhi_TS_Query();

} else {
    // if no metadata option to cancel or continue
    //console.log('aaaw');
    function confirmation() { 
	console.log('cancel or continue...');
	var defer = $.Deferred();

	var $message_NO_metadata_NDVI_MOD13C1005 = $('<div></div>')
	    .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> Function 2/3 (NDVI): No metadata available for NDVI, click OK to continue or cancel to go back to safety')
	    .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
	    title: 'Warning',
	    buttons: {
	        Continue: function () {
	            defer.resolve("true");
	    	    $(this).dialog("close");
                    run_the_ndai_vhi_TS_Query();            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close");
                    $('#ts_ndai_vhi_refresh').attr('src', './js/img/refresh.png');
     	        },
	    },
	    close: function () {
		$(this).remove();
            }
	});

	$message_NO_metadata_NDVI_MOD13C1005.dialog('open').parent().addClass("ui-state-error");
	return defer.promise()
    };

    confirmation().then(function (answer) {
	console.log(answer);
	var ansbool = answer == "true";
	if (ansbool) {
	    console.log('continue as answer confirmation is ' + ansbool); //TRUE
	} else {
	    console.log('return as answer confirmation is ' + ansbool); //FALSE
	    return;
	}
    });  
};

function run_the_ndai_vhi_TS_Query() {

        modal_chart_dialog.dialog("open");//$(".modal_chart").dialog("open");	
		
		//Display right chart
		document.getElementById("nvai-vci-chart").style.display = 'none';
		document.getElementById("ntai-tci-chart").style.display = 'none';
		document.getElementById("ndai-vhi-chart").style.display = 'block';
		document.getElementById("vegetation-chart").style.display = 'none';
		document.getElementById("lst-chart").style.display = 'none';
		document.getElementById("precipitation-chart").style.display = 'none';
		document.getElementById("histogram-chart").style.display = 'none';
		document.getElementById("fracarea-chart").style.display = 'none';

    console.log('lon: ' + lonlat.lon + ', lat: ' + lonlat.lat);
    console.log('dataset: ' + "gpcp");
    console.log('from: ' + ts_start_date);
    console.log('to: ' + ts_end_date);
    //chart_json2 = $('#chart2').highcharts();
    ndai_vhi_chart.showLoading();
    console.log('loading started');

    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecuted_ndai_vhi
    });

    // define the process and append it to OpenLayers.WPS instance
    var ndai_vhi_TS_Process = new OpenLayers.WPS.Process({
        identifier: "WPS_NDAI_VHI_TS",
        inputs: [
            new OpenLayers.WPS.LiteralPut({
                identifier: "from_date",
                value: ts_start_date
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "to_date",
                value: ts_end_date
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "lon_center",
                value: lonlat.lon.toFixed(4)
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "lat_center",
                value: lonlat.lat.toFixed(4)
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "pix_offset",
                value: OffsetPixel
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "pyhants",
                value: document.getElementById("pyhants_yes_no_ndai_vhi").value,
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "ann_freq",
                value: document.getElementById("number_of_frequencies_ndai_vhi").value,
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "outliers_rj",
                value: document.getElementById("type_of_outliers_ndai_vhi").value,
            })
        ],
        async: false,
        outputs: [
            new OpenLayers.WPS.ComplexPut({
                identifier: "ndai_ts",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),
            new OpenLayers.WPS.ComplexPut({
                identifier: "vhi_ts",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),
        ]
    });

    // defined earlier
    wps.addProcess(ndai_vhi_TS_Process);

    // run Execute
    wps.execute("WPS_NDAI_VHI_TS");

    var cache = [];
    console.log(JSON.stringify(ndai_vhi_TS_Process, function(key, value) {
        if (typeof value === 'object' && value !== null) {
            if (cache.indexOf(value) !== -1) {
                // Circular reference found, discard key
                return;
            }
            // Store value in our collection
            cache.push(value);
        }
        return value;
    }));
    cache = null; // Enable garbage collection        


};

};

// Everything went OK 
function onExecuted_ndai_vhi(process) {
    var result0 = process.outputs[0].getValue()
    resu0 = result0.slice(16);
    series0 = '..'
    series0 += resu0

    var result1 = process.outputs[1].getValue()
    resu1 = result1.slice(16);
    series1 = '..'
    series1 += resu1


    //$('#result_url1').text(result1);
    //$('#result_url2').text(result2);  
    console.log('function NDAI VHI works: '+series1);

    //OpenLayers.Util.getElement("result").innerHTML = resu1;
    

    //var chart = $('#chart1').highcharts();

    $.getJSON(series0, function(data_ts) {
        ndai_vhi_chart.series[0].update({
            data: data_ts
        })
    });
    $.getJSON(series1, function(data_ts) {
        ndai_vhi_chart.series[1].update({
            data: data_ts
        })
    });

    // refresh button finished loading
    $('#ts_ndai_vhi_refresh').attr('src', './js/img/refresh.png');
    ndai_vhi_chart.hideLoading();
    ndai_vhi_chart.reflow();
};

// ------- // 

// refresh button change to load
    $('#ts_ndai_vhi_refresh').on('click', function () {
    if ($('#ts_ndai_vhi').is(':checked')) {
        console.log('ts_ndai_vhi_check');
        $('#ts_ndai_vhi_refresh').attr('src', './js/img/ls.gif');
        execute_ndai_vhi_Query(lonlat);        
    } else {
        console.log('ts_ndai_vhi_uncheck');
    }
    });

// ------- // ------- // ------- // ------- // ------- // -------
