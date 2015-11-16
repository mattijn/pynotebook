/**
 * ggplot2 theme for Highcharts JS
 * @author Mattijn van Hoek
 */


Highcharts.theme = {
   colors: ["#7cb5ec", "#f7a35c", "#90ee7e", "#7798BF", "#aaeeee", "#ff0066", "#eeaaee",
      "#55BF3B", "#DF5353", "#7798BF", "#aaeeee"],
   chart: {
      backgroundColor: null,
      plotBackgroundColor: '#F2F2F2',
      style: {
         fontFamily: "arial, sans-serif"
      },
      backgroundColor: 'rgb(255,255,255)',
      zoomType: 'x',
      resetZoomButton: {
          theme: {
             fill: 'white',
             stroke: '#7F7F7F',
             r: 0,
             states: {
                 hover: {
                     fill: '#7F7F7F',
                     style: {
                         color: 'white'
                     }
                 }
             }
          }
      }
   },
   subtitle: {
      style: {
         fontSize: '16px',
         //fontWeight: 'bold',
         //textTransform: 'uppercase'
      }
   },
   tooltip: {
      borderWidth: 0,
      backgroundColor: 'rgba(219,219,216,0.8)',
      shadow: false
   },
   legend: {
      itemStyle: {
         fontWeight: 'bold',
         fontSize: '13px'
      }
   },
   xAxis: {
      type: 'datetime',
      minPadding: 0,
      maxPadding: 0,

      minorGridLineColor: '#FFF',
      minorGridLineWidth: 0.5,
      minorTickLength: 0,
      minorTickInterval: 'auto',
	
      gridLineColor: '#FFF',
      gridLineWidth: 2,
      lineWidth: null,
      tickWidth: 2,
      tickColor: '#7F7F7F',

      labels: {
         style: {
            fontSize: '12px'
         }
      }
   },
   yAxis: {
      minorGridLineColor: '#FFF',
      minorGridLineWidth: 0.5,
      minorTickLength: 0,
      minorTickInterval: 'auto',

      gridLineColor: '#FFF',
      gridLineWidth: 2,
      tickWidth: 2,
      tickColor: '#7F7F7F',

      title: {
         style: {
            textTransform: 'uppercase'
         },
         enabled:true,                    
         align: 'high',
         offset: 10,
         rotation: 0,
         y: -10
      },
      labels: {
         style: {
            fontSize: '12px'
         }
      }
   },
   title: {
       text: null,
   },

   scrollbar: {
       enabled: false
   },		
   loading: {
       labelStyle: {
           color: 'white'
       },
   style: {
       backgroundColor: 'gray'
       }
   },
   credits: {
       enabled: false
   },
   plotOptions: {
      //candlestick: {
      //   lineColor: '#404048'
      //},
   series: {
       marker: {
          enabled: false,
          radius: 2,
       },
       pointWidth: 20
   }
   },


   // General
   background2: '#F0F0EA'
   
};

// Apply the theme
Highcharts.setOptions(Highcharts.theme);





$(function() {
    // Settings Init Vegetation Chart
    drought_derived_chart = new Highcharts.Chart({
        chart:    {renderTo: 'drought-derived-chart'},
	yAxis:    [{title:{text:'Drought Index'},
		    min:-1.5,
		    max:1.5
		}, {title:{text: 'PAP'},
		    min:-100,
		    max: 100,
		    opposite: true
		}],
	subtitle: {text: 'Drought Derived Timeseries'},		
	series:   [{
                    name: 'PAP',
                    data: [],
		    yAxis: 1,
                    tooltip: {valueDecimals: 2},
                    color: "#00C8F0",
                    negativeColor: "#FFAA00",
		    type: 'column'
                  }, {
                    name: 'VCI',
                    data: [],
                    tooltip: {valueDecimals: 2},
                    //color: "#00C8F0",
                    //negativeColor: "#FFAA00"
                  }, {
                    name: 'TCI',
                    data: [],
                    tooltip: {valueDecimals: 2},
                    //color: "#00C8F0",
                    //negativeColor: "#FFAA00"
                  }, {
                    name: 'VHI',
                    data: [],
                    tooltip: {valueDecimals: 2},
                    //color: "#00C8F0",
                    //negativeColor: "#FFAA00"
                  }, {
                    name: 'NVAI',
                    data: [],
                    tooltip: {valueDecimals: 2},
                    //color: "#00C8F0",
                    //negativeColor: "#FFAA00"
                  }, {
                    name: 'NTAI',
                    data: [],
                    tooltip: {valueDecimals: 2},
                    //color: "#00C8F0",
                    //negativeColor: "#FFAA00"
                  }]
    });
	
	
    // Settings Init Vegetation Chart
    hydrology_chart = new Highcharts.Chart({
        chart:    {renderTo: 'hydrology-chart'},
	yAxis:    {title: {text: 'ET'}},
	subtitle: {text: 'Hydrology Timeseries'},		
	series:   [{
                    name: 'ET - MODIS',
                    data: [],
                    tooltip: {
                    valueDecimals: 2}
                  },{
                    name: 'ET - RADI',
                    data: [],
                    tooltip: {valueDecimals: 2},
                  }]
    });
	
	
    // Settings Init Vegetation Chart
    vegetation_chart = new Highcharts.Chart({
        chart:    {renderTo: 'vegetation-chart'},
	yAxis:    {title: {text: 'NDVI'}},
	subtitle: {text: 'Vegetation Timeseries'},
        series:   [{
	            name: 'NDVI',
	            data: [],
	            tooltip: {valueDecimals: 2},
	          }, {
	            name: 'NDVI reconstructed',
	            data: [],
	            tooltip: {valueDecimals: 2},
	          }]
    });
			

    // Settings Init Vegetation Chart	
    meteorology_chart = new Highcharts.Chart({
        chart:    {renderTo: 'meteorology-chart'},
	yAxis:    {title: {text: 'LST'}},
	subtitle: {text: 'Meteorology Timeseries'},		
	series:   [{
                    name: 'LST',
                    data: [],
                    tooltip: {
                    valueDecimals: 2}
                  },{
                    name: 'LST reconstructed',
                    data: [],
                    tooltip: {valueDecimals: 2},
                  }]
    });
	


    // Settings Init Histogram Chart	
    histogram_chart = new Highcharts.Chart({
        chart:    {renderTo: 'histogram-chart'},
	yAxis:    {title: {text: 'frequency'}},
	xAxis:    {type: 'linear'},
	subtitle: {text: 'Histogram Timeseries'},		
	series:   [{
                    name: 'Histogram ts1',
                    data: [],
                    tooltip: {valueDecimals: 2}
                  },{
                    name: 'Histogram ts2',
                    data: [],
                    tooltip: {valueDecimals: 2}
                  },{
                    name: 'Histogram ts3',
                    data: [],
                    tooltip: {valueDecimals: 2}
                  },{
                    name: 'Histogram ts4',
                    data: [],
                    tooltip: {valueDecimals: 2}
                  },{
                    name: 'Histogram ts5',
                    data: [],
                    tooltip: {valueDecimals: 2}
                  }]
    });



    // Settings Init Fracarea Chart	
    fracarea_chart = new Highcharts.Chart({
        chart:    {renderTo: 'fracarea-chart'},
	yAxis:    {title: {text: 'frac. area(%)'},
		   min: 0,
		   max: 100},
	xAxis:    {type: 'linear',
                   min: -1,
		   max: 1},
	subtitle: {text: 'Fractional area Timeseries in %'},		
	series:   [{
                    name: 'Fracarea ts1',
                    data: [],
                    tooltip: {valueDecimals: 2}
                  },{
                    name: 'Fracarea ts2',
                    data: [],
                    tooltip: {valueDecimals: 2}
                  },{
                    name: 'Fracarea ts3',
                    data: [],
                    tooltip: {valueDecimals: 2}
                  },{
                    name: 'Fracarea ts4',
                    data: [],
                    tooltip: {valueDecimals: 2}
                  },{
                    name: 'Fracarea ts5',
                    data: [],
                    tooltip: {valueDecimals: 2}
                  }]
    });

});





// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// Execute Drought Query using the coordinates and parameter setting through PyWPS
function executeDroughtQuery(lonlat) {

        document.getElementById("ts_drought_options").style.display = 'block';

        document.getElementById("ts_drought_refresh").style.display = 'block';
        document.getElementById("ts_hydro_refresh").style.display = 'none';
        document.getElementById("ts_vegetation_refresh").style.display = 'none';
        document.getElementById("ts_meteo_refresh").style.display = 'none';

        document.getElementById("ts_hydro_options").style.display = 'none';
        document.getElementById("ts_vegetation_options").style.display = 'none';
        document.getElementById("ts_meteo_options").style.display = 'none';

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
        .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> Function 1/2: The start date is after the end date!')
        .dialog({
        dialogClass: "no-close",
        modal: true,
        resizable: false,
        title: 'Warning',
        buttons: {
            OK: function () {
                $(this).dialog("close");
                $message_startDate_after_endDate.remove();
                $('#ts_drought_refresh').attr('src', './js/img/refresh.png');
            }
        }
    });

    $message_startDate_after_endDate.dialog('open').parent().addClass("ui-state-error");

    return
};

// check if start_date is after the available data according to metadata
// check if dataset is available based on metadata
if (typeof metadata_gpcp_to_date !== 'undefined') {
    // metadata_gpcp_to_date is defined
    if (ts_start_date > metadata_gpcp_to_date) {
        // your code here.
        var $message_metadata_gpcp = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> Datasource 1/3 (Precipitation): The start date is after the available date! Dataset availability: ' + metadata_gpcp_from_date + ' to ' + metadata_gpcp_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_gpcp.remove();
                    $('#ts_drought_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_gpcp.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    run_the_PAP_TS_Query();

} else {
    // if no metadata option to cancel or continue
    //console.log('aaaw');
    function confirmation() { 
	console.log('cancel or continue...');
	var defer = $.Deferred();

	var $message_NO_metadata_gpcp = $('<div></div>')
	    .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> Function 1/3 (Precipitation): No metadata available, click OK to continue or cancel to go back to safety')
	    .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
	    title: 'Warning',
	    buttons: {
	        Continue: function () {
	            defer.resolve("true");
	    	    $(this).dialog("close");
		    run_the_PAP_TS_Query();
            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close");
                    $('#ts_drought_refresh').attr('src', './js/img/refresh.png');
     	        },
	    },
	    close: function () {
		$(this).remove();
            }
	});

	$message_NO_metadata_gpcp.dialog('open').parent().addClass("ui-state-error");
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

//    if (ts_start_date > ts_end_date) {
//        alert("The start date is after the end date!");
//	$('#ts_drought_refresh').attr('src', './js/img/refresh.png');
//        //$(".modal_chart").dialog("close");
//
//	return;
//    };

//    if (ts_start_date > metadata_gpcp_to_date) {
//        alert("The start date is after the available date! Dataset availability: "+ metadata_gpcp_from_date +' to '+ metadata_gpcp_to_date);
//	$('#ts_drought_refresh').attr('src', './js/img/refresh.png');
//        //$(".modal_chart").dialog("close");
//	return;
//    };

function run_the_PAP_TS_Query() {



        modal_chart_dialog.dialog("open");//$(".modal_chart").dialog("open");			
		//Display right chart
		document.getElementById("drought-derived-chart").style.display = 'block';
		document.getElementById("hydrology-chart").style.display = 'none';
		document.getElementById("vegetation-chart").style.display = 'none';
		document.getElementById("meteorology-chart").style.display = 'none';
		document.getElementById("histogram-chart").style.display = 'none';
		document.getElementById("fracarea-chart").style.display = 'none';

    console.log('lon: ' + lonlat.lon + ', lat: ' + lonlat.lat);
    console.log('dataset: ' + "gpcp");
    console.log('from: ' + ts_start_date);
    console.log('to: ' + ts_end_date);
    //chart_json2 = $('#chart2').highcharts();
    drought_derived_chart.showLoading();
    console.log('loading started');

    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecutedDroughtDerived
    });

    // define the process and append it to OpenLayers.WPS instance
    var PAP_TS_Process = new OpenLayers.WPS.Process({
        identifier: "WPS_PRECIP_DI_CAL_TS",
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
                identifier: "pap_lon",
                value: lonlat.lon.toFixed(4)
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "pap_lat",
                value: lonlat.lat.toFixed(4)
            })
        ],
        async: false,
        outputs: [
            new OpenLayers.WPS.ComplexPut({
                identifier: "pap_ts",
                asReference: true
                    //formats:{mimeType:'application/json'}
            })
        ]
    });

    // defined earlier
    wps.addProcess(PAP_TS_Process);

    // run Execute
    wps.execute("WPS_PRECIP_DI_CAL_TS");

    var cache = [];
    console.log(JSON.stringify(PAP_TS_Process, function(key, value) {
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

/**

 * WPS events
 */
// Everything went OK 
function onExecutedDroughtDerived(process) {
    var result1 = process.outputs[0].getValue()

    resu1 = result1.slice(16);

    series1 = '..'
    series1 += resu1


    //$('#result_url1').text(result1);
    //$('#result_url2').text(result2);  
    console.log('function Drought derived works: '+series1);

    //OpenLayers.Util.getElement("result").innerHTML = resu1;
    

    //var chart = $('#chart1').highcharts();

    $.getJSON(series1, function(data_ts) {
        drought_derived_chart.series[0].update({
            data: data_ts
        })
    });

    // refresh button finished loading
    $('#ts_drought_refresh').attr('src', './js/img/refresh.png');
    //drought_derived_chart.hideLoading();
    //drought_derived_chart.reflow();


};

// ------- // 

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// Execute ALLDI Query using the coordinates and parameter setting through PyWPS
function executeALLDIQuery(lonlat) {
        document.getElementById("ts_drought_options").style.display = 'block';

        document.getElementById("ts_drought_refresh").style.display = 'block';
        document.getElementById("ts_hydro_refresh").style.display = 'none';
        document.getElementById("ts_vegetation_refresh").style.display = 'none';
        document.getElementById("ts_meteo_refresh").style.display = 'none';

        document.getElementById("ts_hydro_options").style.display = 'none';
        document.getElementById("ts_vegetation_options").style.display = 'none';
        document.getElementById("ts_meteo_options").style.display = 'none';

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
                $('#ts_drought_refresh').attr('src', './js/img/refresh.png');
            }
        }
    });

    $message_startDate_after_endDate.dialog('open').parent().addClass("ui-state-error");

    return
};

// check if start_date is after the available data according to metadata
// check if dataset is available based on metadata
if (typeof metadata_NDVI_MOD13C1005_to_date !== 'undefined') {
    // metadata_gpcp_to_date is defined
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
                    $('#ts_drought_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_NDVI_MOD13C1005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    //run_the_PAP_TS_Query();

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
		    //run_the_PAP_TS_Query();
            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close");
                    $('#ts_drought_refresh').attr('src', './js/img/refresh.png');
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

// check if start_date is after the available data according to metadata
// check if dataset is available based on metadata
if (typeof metadata_LST_MOD11C2005_to_date !== 'undefined') {
    // metadata_gpcp_to_date is defined
    if (ts_start_date > metadata_LST_MOD11C2005_to_date) {
        // your code here.
        var $message_metadata_LST_MOD11C2005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> Function 3/3 (LST): The start date is after the available date! Dataset availability: ' + metadata_LST_MOD11C2005_from_date + ' to ' + metadata_LST_MOD11C2005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_LST_MOD11C2005.remove();
                }
            }
        });
        $message_metadata_LST_MOD11C2005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    run_the_ALLDI_TS_Query();

} else {
    // if no metadata option to cancel or continue
    //console.log('aaaw');
    function confirmation() { 
	console.log('cancel or continue...');
	var defer = $.Deferred();

	var $message_NO_metadata_LST_MOD11C2005 = $('<div></div>')
	    .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> Function 3/3 (LST): No metadata available for LST, click OK to continue or cancel to go back to safety')
	    .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
	    title: 'Warning',
	    buttons: {
	        Continue: function () {
	            defer.resolve("true");
	    	    $(this).dialog("close");
		    run_the_ALLDI_TS_Query();
            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close");
                    $('#ts_drought_refresh').attr('src', './js/img/refresh.png');
     	        },
	    },
	    close: function () {
		$(this).remove();
            }
	});

	$message_NO_metadata_LST_MOD11C2005.dialog('open').parent().addClass("ui-state-error");
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

//    if (ts_start_date > ts_end_date) {
//        alert("The start date is after the end date!");
//	$('#ts_drought_refresh').attr('src', './js/img/refresh.png');
//        //$(".modal_chart").dialog("close");
//
//	return;
//    };

//    if (ts_start_date > metadata_modis_11c2_cov_to_date) {
//        alert("The start date is after the available date! Dataset availability: "+ metadata_modis_11c2_cov_from_date +' to '+ metadata_modis_11c2_cov_to_date);
//	$('#ts_drought_refresh').attr('src', './js/img/refresh.png');
//        //$(".modal_chart").dialog("close");
//	return;
//    };

//    if (ts_start_date > metadata_modis_13c1_cov_to_date) {
//        alert("The start date is after the available date! Dataset availability: "+ metadata_modis_13c1_cov_from_date +' to '+ metadata_modis_13c1_cov_to_date);
//	$('#ts_drought_refresh').attr('src', './js/img/refresh.png');
//        //$(".modal_chart").dialog("close");
//	return;
//    };

function run_the_ALLDI_TS_Query() {



        modal_chart_dialog.dialog("open");//$(".modal_chart").dialog("open");			
		//Display right chart
		document.getElementById("drought-derived-chart").style.display = 'block';
		document.getElementById("hydrology-chart").style.display = 'none';
		document.getElementById("vegetation-chart").style.display = 'none';
		document.getElementById("meteorology-chart").style.display = 'none';
		document.getElementById("histogram-chart").style.display = 'none';
		document.getElementById("fracarea-chart").style.display = 'none';

    console.log('lon: ' + lonlat.lon + ', lat: ' + lonlat.lat);
    console.log('dataset: ' + "gpcp");
    console.log('from: ' + ts_start_date);
    console.log('to: ' + ts_end_date);
    //chart_json2 = $('#chart2').highcharts();
    drought_derived_chart.showLoading();
    console.log('loading started');

    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecutedALLDI
    });

    // define the process and append it to OpenLayers.WPS instance
    var ALLDI_TS_Process = new OpenLayers.WPS.Process({
        identifier: "WPS_ALL_DI_CAL_TS",
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
                value: document.getElementById("pyhants_yes_no_nvai").value,
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "ann_freq",
                value: document.getElementById("number_of_frequencies_nvai").value,
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "outliers_rj",
                value: document.getElementById("type_of_outliers_nvai").value,
            })
        ],
        async: false,
        outputs: [
            new OpenLayers.WPS.ComplexPut({
                identifier: "vci_ts",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),
            new OpenLayers.WPS.ComplexPut({
                identifier: "tci_ts",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),
            new OpenLayers.WPS.ComplexPut({
                identifier: "vhi_ts",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),
            new OpenLayers.WPS.ComplexPut({
                identifier: "nvai_ts",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),
            new OpenLayers.WPS.ComplexPut({
                identifier: "ntai_ts",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),
        ]
    });

    // defined earlier
    wps.addProcess(ALLDI_TS_Process);

    // run Execute
    wps.execute("WPS_ALL_DI_CAL_TS");

    var cache = [];
    console.log(JSON.stringify(ALLDI_TS_Process, function(key, value) {
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

/**


 * WPS events
 */
// Everything went OK 
function onExecutedALLDI(process) {
    var result0 = process.outputs[0].getValue()
    resu0 = result0.slice(16);
    series0 = '..'
    series0 += resu0

    var result1 = process.outputs[1].getValue()
    resu1 = result1.slice(16);
    series1 = '..'
    series1 += resu1

    var result2 = process.outputs[2].getValue()
    resu2 = result2.slice(16);
    series2 = '..'
    series2 += resu2

    var result3 = process.outputs[3].getValue()
    resu3 = result3.slice(16);
    series3 = '..'
    series3 += resu3

    var result4 = process.outputs[4].getValue()
    resu4 = result4.slice(16);
    series4 = '..'
    series4 += resu4

    //$('#result_url1').text(result1);
    //$('#result_url2').text(result2);  
    console.log('function NVAI works: '+series1);

    //OpenLayers.Util.getElement("result").innerHTML = resu1;
    

    //var chart = $('#chart1').highcharts();

    $.getJSON(series0, function(data_ts) {
        drought_derived_chart.series[1].update({
            data: data_ts
        })
    });
    $.getJSON(series1, function(data_ts) {
        drought_derived_chart.series[2].update({
            data: data_ts
        })
    });
    $.getJSON(series2, function(data_ts) {
        drought_derived_chart.series[3].update({
            data: data_ts
        })
    });
    $.getJSON(series3, function(data_ts) {
        drought_derived_chart.series[4].update({
            data: data_ts
        })
    });
    $.getJSON(series4, function(data_ts) {
        drought_derived_chart.series[5].update({
            data: data_ts
        })
    });

    // refresh button finished loading
    $('#ts_drought_refresh').attr('src', './js/img/refresh.png');
    drought_derived_chart.hideLoading();
    drought_derived_chart.reflow();


};

// ------- // 

// refresh button change to load
    $('#ts_drought_refresh').on('click', function () {
    if ($('#ts_drought').is(':checked')) {
        console.log('ts_drought_check');
        $('#ts_drought_refresh').attr('src', './js/img/ls.gif');
        executeDroughtQuery(lonlat);
        //executeNVAIQuery(lonlat);
        executeALLDIQuery(lonlat);
    } else {
        console.log('ts_drought_uncheck');
    }
    });

// ------- // ------- // ------- // ------- // ------- // -------


// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// Execute NVAI Query using the coordinates and parameter setting through PyWPS
/*
function executeNVAI_TS_Query(lonlat) {
        document.getElementById("ts_drought_options").style.display = 'none';
        document.getElementById("ts_hydro_options").style.display = 'block';

        document.getElementById("ts_drought_refresh").style.display = 'none';
        document.getElementById("ts_hydro_refresh").style.display = 'block';
        document.getElementById("ts_vegetation_refresh").style.display = 'none';
        document.getElementById("ts_meteo_refresh").style.display = 'none';

        document.getElementById("ts_vegetation_options").style.display = 'none';
        document.getElementById("ts_meteo_options").style.display = 'none';

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

    if (ts_start_date > ts_end_date) {
        alert("The start date is after the end date!");
	$('#ts_drought_refresh').attr('src', './js/img/refresh.png');
        //$(".modal_chart").dialog("close");

	return;
    };

//    if (ts_start_date > metadata_modis_11c2_cov_to_date) {
//        alert("The start date is after the available date! Dataset availability: "+ metadata_modis_11c2_cov_from_date +' to '+ metadata_modis_11c2_cov_to_date);
//	$('#ts_drought_refresh').attr('src', './js/img/refresh.png');
//        //$(".modal_chart").dialog("close");
//	return;
//    };






        modal_chart_dialog.dialog("open");//$(".modal_chart").dialog("open");
		
		//Display right chart
		document.getElementById("drought-derived-chart").style.display = 'none';
		document.getElementById("hydrology-chart").style.display = 'block';
		document.getElementById("vegetation-chart").style.display = 'none';
		document.getElementById("meteorology-chart").style.display = 'none';
		document.getElementById("histogram-chart").style.display = 'none';
		document.getElementById("fracarea-chart").style.display = 'none';

    console.log('lon: ' + lonlat.lon + ', lat: ' + lonlat.lat);
    console.log('dataset: ' + "gpcp");
    console.log('from: ' + ts_start_date);
    console.log('to: ' + ts_end_date);
    //chart_json2 = $('#chart2').highcharts();
    drought_derived_chart.showLoading();
    console.log('loading started');

    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecutedNVAI_TS
    });

    // define the process and append it to OpenLayers.WPS instance
    var NVAI_TS_Process = new OpenLayers.WPS.Process({
        identifier: "WPS_NVAI_DI_CAL_TS",
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
                value: document.getElementById("pyhants_yes_no_nvai").value,
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "ann_freq",
                value: document.getElementById("number_of_frequencies_nvai").value,
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "outliers_rj",
                value: document.getElementById("type_of_outliers_nvai").value,
            })
        ],
        async: false,
        outputs: [
            new OpenLayers.WPS.ComplexPut({
                identifier: "nvai_ts",
                asReference: true
                    //formats:{mimeType:'application/json'}
            })
        ]
    });

    // defined earlier
    wps.addProcess(NVAI_TS_Process);

    // run Execute
    wps.execute("WPS_NVAI_DI_CAL_TS");

    var cache = [];
    console.log(JSON.stringify(NVAI_TS_Process, function(key, value) {
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

*/

/**

 * WPS events
 */
// Everything went OK 
/*
function onExecutedNVAI_TS(process) {
    var result1 = process.outputs[0].getValue()

    resu1 = result1.slice(16);

    series1 = '..'
    series1 += resu1


    //$('#result_url1').text(result1);
    //$('#result_url2').text(result2);  
    console.log('function NVAI works: '+series1);

    //OpenLayers.Util.getElement("result").innerHTML = resu1;
    

    //var chart = $('#chart1').highcharts();

    $.getJSON(series1, function(data_ts) {
        drought_derived_chart.series[1].update({
            data: data_ts
        })
    });

    // refresh button finished loading
    $('#ts_drought_refresh').attr('src', './js/img/refresh.png');
    drought_derived_chart.hideLoading();
    drought_derived_chart.reflow();


};

// ------- // 

// refresh button change to load
//    $('#ts_drought_refresh').on('click', function () {
//    if ($('#ts_drought').is(':checked')) {
//        console.log('ts_drought_check');
//        $('#ts_drought_refresh').attr('src', './js/img/ls.gif');
//        executeDroughtQuery(lonlat);
//        executeNVAIQuery(lonlat);
//    } else {
//        console.log('ts_drought_uncheck');
//    }
//    });
//
// ------- // ------- // ------- // ------- // ------- // -------
*/

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// Execute Vegetation Query using the coordinates and parameter setting through PyWPS
function executeVegetationQuery(lonlat) {

        document.getElementById("ts_drought_options").style.display = 'none';
        document.getElementById("ts_hydro_options").style.display = 'none';
        document.getElementById("ts_vegetation_options").style.display = 'block';

        document.getElementById("ts_drought_refresh").style.display = 'none';
        document.getElementById("ts_hydro_refresh").style.display = 'none';
        document.getElementById("ts_vegetation_refresh").style.display = 'block';
        document.getElementById("ts_meteo_refresh").style.display = 'none';

        document.getElementById("ts_meteo_options").style.display = 'none';

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
        .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The start date is after the end date!')
        .dialog({
        dialogClass: "no-close",
        modal: true,
        resizable: false,
        title: 'Warning',
        buttons: {
            OK: function () {
                $(this).dialog("close");
                $message_startDate_after_endDate.remove();
                $('#ts_vegetation_refresh').attr('src', './js/img/refresh.png');
            }
        }
    });

    $message_startDate_after_endDate.dialog('open').parent().addClass("ui-state-error");

    return
};

// check if start_date is after the available data according to metadata
// check if dataset is available based on metadata
if (typeof metadata_NDVI_MOD13C1005_to_date !== 'undefined') {
    // metadata_gpcp_to_date is defined
    if (ts_start_date > metadata_NDVI_MOD13C1005_to_date) {
        // your code here.
        var $message_metadata_NDVI_MOD13C1005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The start date is after the available date! Dataset availability: ' + metadata_NDVI_MOD13C1005_from_date + ' to ' + metadata_NDVI_MOD13C1005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_NDVI_MOD13C1005.remove();
                    $('#ts_vegetation_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_NDVI_MOD13C1005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    run_the_Vegetation_TS_Query();

} else {
    // if no metadata option to cancel or continue
    //console.log('aaaw');
    function confirmation() { 
	console.log('cancel or continue...');
	var defer = $.Deferred();

	var $message_NO_metadata_NDVI_MOD13C1005 = $('<div></div>')
	    .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No metadata available, click OK to continue or cancel to go back to safety')
	    .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
	    title: 'Warning',
	    buttons: {
	        Continue: function () {
	            defer.resolve("true");
	    	    $(this).dialog("close");
		    run_the_Vegetation_TS_Query();
            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close");
                    $('#ts_vegetation_refresh').attr('src', './js/img/refresh.png');
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





/*
    if (ts_start_date > ts_end_date) {
        alert("The start date is after the end date!");
        $('#ts_vegetation_refresh').attr('src', './js/img/refresh.png');
        //$(".modal_chart").dialog("close");
	return;
    };

    if (ts_start_date > metadata_modis_11c2_cov_to_date) {
        alert("The start date is after the available dates! Dataset availability: "+ metadata_modis_11c2_cov_from_date +' to '+ metadata_modis_11c2_cov_to_date);
        $('#ts_vegetation_refresh').attr('src', './js/img/refresh.png');
        //$(".modal_chart").dialog("close");
	return;
    };
*/

function run_the_Vegetation_TS_Query() {



        modal_chart_dialog.dialog("open");//$(".modal_chart").dialog("open");
		
		//Display right chart
		document.getElementById("drought-derived-chart").style.display = 'none';
		document.getElementById("hydrology-chart").style.display = 'none';
		document.getElementById("vegetation-chart").style.display = 'block';
		document.getElementById("meteorology-chart").style.display = 'none';
		document.getElementById("histogram-chart").style.display = 'none';
		document.getElementById("fracarea-chart").style.display = 'none';

    console.log('lon: ' + lonlat.lon + ', lat: ' + lonlat.lat);
    console.log('dataset: ' + "modis_13c1_cov");
    console.log('from: ' + ts_start_date);
    console.log('to: ' + ts_end_date);
    //chart_json2 = $('#chart2').highcharts();
    vegetation_chart.showLoading();
    console.log('loading started');

    //var urlRA2 = 'http://localhost:8080/rasdaman/ows/wcs2?service=WCS&version=2.0.1&request=GetCoverage&coverageId=' + "modis_13c1_cov" + '&subset=Long(' + lonlat.lon + ',' + lonlat.lon + ')&subset=Lat(' + lonlat.lat + ',' + lonlat.lat + ')&subset=ansi(%22' + ts_start_date + '%22,%22' + ts_end_date + '%22)';
    //console.log('total url: ' + urlRA2);
    // set the proxy
    // OpenLayers.ProxyHost = "/cgi-bin/proxy.cgi?url=";

    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecutedVegetation
    });

    // define the process and append it to OpenLayers.WPS instance
    var PyHANTS_Process = new OpenLayers.WPS.Process({
        identifier: "pyhants_3",
        inputs: [
            new OpenLayers.WPS.LiteralPut({
                identifier: "covID",
                value: 'NDVI_MOD13C1005'
            }),
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
                identifier: "no_freq",
                value: document.getElementById("number_of_frequencies").value,
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "outlier_type",
                value: document.getElementById("type_of_outliers").value,
            })
        ],
        async: false,
        outputs: [
            new OpenLayers.WPS.ComplexPut({
                identifier: "output_ts1",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),
            new OpenLayers.WPS.ComplexPut({
                identifier: "output_ts2",
                asReference: true
                    //formats:{mimeType:'application/json'}
            })
        ]
    });

    // defined earlier
    wps.addProcess(PyHANTS_Process);

    // run Execute
    wps.execute("pyhants_3");

    var cache = [];
    console.log(JSON.stringify(PyHANTS_Process, function(key, value) {
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

/**
 * WPS events
 */
// Everything went OK 
function onExecutedVegetation(process) {
    var result1 = process.outputs[0].getValue()
    var result2 = process.outputs[1].getValue()
    resu1 = result1.slice(16);
    resu2 = result2.slice(16);

    series1 = '..'
    series1 += resu1
    series2 = '..'
    series2 += resu2

    //$('#result_url1').text(result1);
    //$('#result_url2').text(result2);  
    console.log('function Vegetation works: ' + series2);

    //OpenLayers.Util.getElement("result").innerHTML = resu1;
    

    //var chart = $('#chart1').highcharts();

    $.getJSON(series2, function(data_ts) {
        vegetation_chart.series[0].update({
            data: data_ts
        })
    });
    $.getJSON(series1, function(data_ts) {
        vegetation_chart.series[1].update({
            data: data_ts
        });
    });

    // refresh button finished loading
    $('#ts_vegetation_refresh').attr('src', './js/img/refresh.png');
    vegetation_chart.hideLoading();
    vegetation_chart.reflow();


};

// ------- // 

// refresh button change to load
    $('#ts_vegetation_refresh').on('click', function () {
    if ($('#ts_vegetation').is(':checked')) {
        console.log('ts_vegetation_check');
        $('#ts_vegetation_refresh').attr('src', './js/img/ls.gif');
        executeVegetationQuery(lonlat);
    } else {
        console.log('ts_vegetation_uncheck');
    }
    });

// ------- // ------- // ------- // ------- // ------- // -------


// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
// Execute Meteorology Query using the coordinates and parameter setting through PyWPS
function executeMeteorologyQuery(lonlat) {

        document.getElementById("ts_drought_options").style.display = 'none';
        document.getElementById("ts_hydro_options").style.display = 'none';
        document.getElementById("ts_vegetation_options").style.display = 'none';
        document.getElementById("ts_meteo_options").style.display = 'block';

        document.getElementById("ts_drought_refresh").style.display = 'none';
        document.getElementById("ts_hydro_refresh").style.display = 'none';
        document.getElementById("ts_vegetation_refresh").style.display = 'none';
        document.getElementById("ts_meteo_refresh").style.display = 'block';

    if ($('#startDate_ts').length > 0) {
        // dostuff
        ts_start_date = $('#startDate_ts').MonthPicker('GetSelectedYear') + '-' + pad($(
            '#startDate_ts').MonthPicker('GetSelectedMonth')) + '-' + '01';
    } else {
        ts_start_date = '2009-01-01';
        console.log('date_in not undefined, set to default:');
    };
	
    if ($('#endDate_ts').length > 0) {
        // dostuff
        ts_end_date = $('#endDate_ts').MonthPicker('GetSelectedYear') + '-' + pad($(
            '#endDate_ts').MonthPicker('GetSelectedMonth')) + '-' + '01';
    } else {
        ts_end_date = '2011-01-01';
        console.log('date_in not undefined, set to default:');
    };

    if (ts_start_date > ts_end_date) {
        alert("The start date is after the end date!");
	$('#ts_meteo_refresh').attr('src', './js/img/refresh.png');
        //$(".modal_chart").dialog("close");
	return;
    };



// check if start_date is after the end_date
if (ts_start_date > ts_end_date) {
    // your code here.
    var $message_startDate_after_endDate = $('<div></div>')
        .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The start date is after the end date!')
        .dialog({
        dialogClass: "no-close",
        modal: true,
        resizable: false,
        title: 'Warning',
        buttons: {
            OK: function () {
                $(this).dialog("close");
                $message_startDate_after_endDate.remove();
                $('#ts_meteo_refresh').attr('src', './js/img/refresh.png');
            }
        }
    });

    $message_startDate_after_endDate.dialog('open').parent().addClass("ui-state-error");

    return
};

// check if start_date is after the available data according to metadata
// check if dataset is available based on metadata
if (typeof metadata_LST_MOD11C2005_to_date !== 'undefined') {
    // metadata_gpcp_to_date is defined
    if (ts_start_date > metadata_LST_MOD11C2005_to_date) {
        // your code here.
        var $message_metadata_LST_MOD11C2005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The start date is after the available date! Dataset availability: ' + metadata_LST_MOD11C2005_from_date + ' to ' + metadata_LST_MOD11C2005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_LST_MOD11C2005.remove();
                    $('#ts_meteo_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_LST_MOD11C2005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    run_the_LST_TS_Query();

} else {
    // if no metadata option to cancel or continue
    //console.log('aaaw');
    function confirmation() { 
	console.log('cancel or continue...');
	var defer = $.Deferred();

	var $message_NO_metadata_LST_MOD11C2005 = $('<div></div>')
	    .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No metadata available, click OK to continue or cancel to go back to safety')
	    .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
	    title: 'Warning',
	    buttons: {
	        Continue: function () {
	            defer.resolve("true");
	    	    $(this).dialog("close");
		    run_the_LST_TS_Query();
            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close")
                    $('#ts_meteo_refresh').attr('src', './js/img/refresh.png');;
     	        },
	    },
	    close: function () {
		$(this).remove();
            }
	});

	$message_NO_metadata_LST_MOD11C2005.dialog('open').parent().addClass("ui-state-error");
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


//    if (ts_start_date > metadata_modis_13c1_cov_to_date) {
//        alert("The start date is after the available dates! Dataset availability: "+ metadata_modis_13c1_cov_from_date +' to '+ metadata_modis_13c1_cov_to_date);
//	$('#ts_meteo_refresh').attr('src', './js/img/refresh.png');
//        //$(".modal_chart").dialog("close");
//	return;
//    };	

function run_the_LST_TS_Query() {


        modal_chart_dialog.dialog("open");//$(".modal_chart").dialog("open");
		
		//Display right chart
		document.getElementById("drought-derived-chart").style.display = 'none';
		document.getElementById("hydrology-chart").style.display = 'none';
		document.getElementById("vegetation-chart").style.display = 'none';
		document.getElementById("meteorology-chart").style.display = 'block';
		document.getElementById("histogram-chart").style.display = 'none';
		document.getElementById("fracarea-chart").style.display = 'none';	


    console.log('lon: ' + lonlat.lon + ', lat: ' + lonlat.lat);
    console.log('dataset: ' + "modis_11c2_cov");
    console.log('from: ' + ts_start_date);
    console.log('to: ' + ts_end_date);
    //chart_json2 = $('#chart2').highcharts();
    meteorology_chart.showLoading();
    console.log('loading started');

    //var urlRA2 = 'http://localhost:8080/rasdaman/ows/wcs2?service=WCS&version=2.0.1&request=GetCoverage&coverageId=' + "modis_11c2_cov" + '&subset=Long(' + lonlat.lon + ',' + lonlat.lon + ')&subset=Lat(' + lonlat.lat + ',' + lonlat.lat + ')&subset=ansi(%22' + ts_start_date + '%22,%22' + ts_end_date + '%22)';
    //console.log('total url: ' + urlRA2);
    // set the proxy
    // OpenLayers.ProxyHost = "/cgi-bin/proxy.cgi?url=";

    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecutedMeteorology
    });

    // define the process and append it to OpenLayers.WPS instance
    var PyHANTS_Process = new OpenLayers.WPS.Process({
        identifier: "pyhants_3",
        inputs: [
            new OpenLayers.WPS.LiteralPut({
                identifier: "covID",
                value: 'LST_MOD11C2005' 
            }),
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
                identifier: "no_freq",
                value: document.getElementById("number_of_frequencies_meteorology").value,
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "outlier_type",
                value: document.getElementById("type_of_outliers_meteorology").value,
            })
        ],
        async: false,
        outputs: [
            new OpenLayers.WPS.ComplexPut({
                identifier: "output_ts1",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),
            new OpenLayers.WPS.ComplexPut({
                identifier: "output_ts2",
                asReference: true
                    //formats:{mimeType:'application/json'}
            })
        ]
    });

    // defined earlier
    wps.addProcess(PyHANTS_Process);

    // run Execute
    wps.execute("pyhants_3");

    var cache = [];
    console.log(JSON.stringify(PyHANTS_Process, function(key, value) {
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

/**
 * WPS events
 */
// Everything went OK 
function onExecutedMeteorology(process) {
    var result1 = process.outputs[0].getValue()
    var result2 = process.outputs[1].getValue()
    resu1 = result1.slice(16);
    resu2 = result2.slice(16);

    series1 = '..'
    series1 += resu1
    series2 = '..'
    series2 += resu2

    //$('#result_url1').text(result1);
    //$('#result_url2').text(result2); 
    console.log('function Meteorology works: ' + series2);   

    //OpenLayers.Util.getElement("result").innerHTML = resu1;
    

    //var chart = $('#chart1').highcharts();

    $.getJSON(series2, function(data_ts) {
        meteorology_chart.series[0].update({
            data: data_ts
        })
    });
    $.getJSON(series1, function(data_ts) {
        meteorology_chart.series[1].update({
            data: data_ts
        });
    });

    // refresh button finished loading
    $('#ts_meteo_refresh').attr('src', './js/img/refresh.png');
    meteorology_chart.hideLoading();
    meteorology_chart.reflow();


};

// ------- // 

// refresh button change to load
    $('#ts_meteo_refresh').on('click', function () {
    if ($('#ts_meteo').is(':checked')) {
        console.log('ts_meteo_check');
        $('#ts_meteo_refresh').attr('src', './js/img/ls.gif');
        executeMeteorologyQuery(lonlat);
    } else {
        console.log('ts_meteo_uncheck');
    }
    });

// ------- // ------- // ------- // ------- // ------- // -------
