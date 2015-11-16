    // --- // --- // --- // --- // --- // --- // --- // --- //
	
    var drought_derived_chart;
    var hydrology_chart;
	var vegetation_chart;
	var meteorology_chart;
	var lonlat;
    var size = new OpenLayers.Size(40);
    var offset = new OpenLayers.Pixel(-15, -45);
    var icon = new OpenLayers.Icon('./js/img/icon.png', size, offset);

    function open_dialog_event() {
		// Make sure the chart fits the dialog window
		drought_derived_chart.reflow();
		hydrology_chart.reflow();
		vegetation_chart.reflow();
		meteorology_chart.reflow();				
                histogram_chart.reflow();
    };
	
    function close_dialog_event() {
		// Uncheck all radio buttons
		$('.radio_ts_data').prop('checked', false);
	var ts_drought_options = document.getElementById("ts_drought_options");
        if (ts_drought_options) {ts_drought_options.style.display = 'none'};
	
        var ts_hydro_options = document.getElementById("ts_hydro_options");
        if (ts_hydro_options) {ts_hydro_options.style.display = 'none'};

        var ts_vegetation_options = document.getElementById("ts_vegetation_options");
        if (ts_vegetation_options) {ts_vegetation_options.style.display = 'none'};

	var ts_meteo_options = document.getElementById("ts_meteo_options");
        if (ts_meteo_options) {ts_meteo_options.style.display = 'none'};

        var ts_drought_refresh = document.getElementById("ts_drought_refresh");
        if (ts_drought_refresh) {ts_drought_refresh.style.display = 'none'};

	var ts_hydro_refresh = document.getElementById("ts_hydro_refresh");
        if (ts_hydro_refresh) {ts_hydro_refresh.style.display = 'none'};

	var ts_vegetation_refresh = document.getElementById("ts_vegetation_refresh");
        if (ts_vegetation_refresh) {ts_vegetation_refresh.style.display = 'none'};

        var ts_meteo_refresh = document.getElementById("ts_meteo_refresh")
        if (ts_meteo_refresh) {ts_meteo_refresh.style.display = 'none'};


        selector = undefined;	

    };	
	

    $cloned_modal_chart = $('.modal_chart').clone();
    $('.modal_chart').remove();
    var modal_chart_dialog = $cloned_modal_chart.dialog({
        dialogClass: 'fixed-dialog',
        resizable: false,
        autoOpen: false,
        modal: false,
        open: open_dialog_event,
        close: close_dialog_event
    });


/*
    $('.modal_chart').dialog({
    dialogClass: 'fixed-dialog',
    resizable: false,
    autoOpen: false,
    modal: false,
    open: open_dialog_event,
	close: close_dialog_event
    });
*/
		
	
    // --- // --- // --- // --- // --- // --- // --- // --- //


	var ts_drought_options = document.getElementById("ts_drought_options");
        if (ts_drought_options) {ts_drought_options.style.display = 'none'};
	
        var ts_hydro_options = document.getElementById("ts_hydro_options");
        if (ts_hydro_options) {ts_hydro_options.style.display = 'none'};

        var ts_vegetation_options = document.getElementById("ts_vegetation_options");
        if (ts_vegetation_options) {ts_vegetation_options.style.display = 'none'};

	var ts_meteo_options = document.getElementById("ts_meteo_options");
        if (ts_meteo_options) {ts_meteo_options.style.display = 'none'};

        var ts_drought_refresh = document.getElementById("ts_drought_refresh");
        if (ts_drought_refresh) {ts_drought_refresh.style.display = 'none'};

	var ts_hydro_refresh = document.getElementById("ts_hydro_refresh");
        if (ts_hydro_refresh) {ts_hydro_refresh.style.display = 'none'};

	var ts_vegetation_refresh = document.getElementById("ts_vegetation_refresh");
        if (ts_vegetation_refresh) {ts_vegetation_refresh.style.display = 'none'};

        var ts_meteo_refresh = document.getElementById("ts_meteo_refresh")
        if (ts_meteo_refresh) {ts_meteo_refresh.style.display = 'none'};
/*	
    document.getElementById("ts_drought_options").style.display = 'none';
    document.getElementById("ts_hydro_options").style.display = 'none';
    document.getElementById("ts_vegetation_options").style.display = 'none';
    document.getElementById("ts_meteo_options").style.display = 'none';

        document.getElementById("ts_drought_refresh").style.display = 'none';
        document.getElementById("ts_hydro_refresh").style.display = 'none';
        document.getElementById("ts_vegetation_refresh").style.display = 'none';
        document.getElementById("ts_meteo_refresh").style.display = 'none';
	

    var selector;
    $('.radio_ts_data').on('change', function() {
	
	// Check lonlat exist
	if (typeof lonlat == 'undefined') {
		alert('Computer says no. First click on land to retrieve a coordinate.');
		close_dialog_event();
		return;
	};
	
    selector = $('input[name="ts_selector"]:checked', '#ts_form').val()
    if (selector == 'ts_drought') {
        //Add Derived Drought Indices OPTIONS
        console.log('di_options');	
        $('#ts_drought_refresh').attr('src', './js/img/ls.gif');
		
        document.getElementById("ts_drought_options").style.display = 'block';
        document.getElementById("ts_drought_refresh").style.display = 'block';

        document.getElementById("ts_hydro_refresh").style.display = 'none';
        document.getElementById("ts_vegetation_refresh").style.display = 'none';
        document.getElementById("ts_meteo_refresh").style.display = 'none';
        document.getElementById("ts_hydro_options").style.display = 'none';
        document.getElementById("ts_vegetation_options").style.display = 'none';
        document.getElementById("ts_meteo_options").style.display = 'none';
		
        $(".modal_chart").dialog("open");			
		//Display right chart
		document.getElementById("drought-derived-chart").style.display = 'block';
		document.getElementById("hydrology-chart").style.display = 'none';
		document.getElementById("vegetation-chart").style.display = 'none';
		document.getElementById("meteorology-chart").style.display = 'none';
		document.getElementById("histogram-chart").style.display = 'none';
		document.getElementById("fracarea-chart").style.display = 'none';

		// Execute PyWPS to request the data
		executeDroughtQuery(lonlat);
                //executeNVAIQuery(lonlat); 
		executeALLDIQuery(lonlat); 
		
    };
    if (selector == 'ts_hydro') {
        //Add Hydrology OPTIONS
        console.log('hydro_options');
        //$('#ts_hydro_refresh').attr('src', './js/img/ls.gif');

        document.getElementById("ts_drought_options").style.display = 'none';
        document.getElementById("ts_hydro_options").style.display = 'block';

        document.getElementById("ts_drought_refresh").style.display = 'none';
        document.getElementById("ts_hydro_refresh").style.display = 'block';
        document.getElementById("ts_vegetation_refresh").style.display = 'none';
        document.getElementById("ts_meteo_refresh").style.display = 'none';

        document.getElementById("ts_vegetation_options").style.display = 'none';
        document.getElementById("ts_meteo_options").style.display = 'none';
        $(".modal_chart").dialog("open");
		
		//Display right chart
		document.getElementById("drought-derived-chart").style.display = 'none';
		document.getElementById("hydrology-chart").style.display = 'block';
		document.getElementById("vegetation-chart").style.display = 'none';
		document.getElementById("meteorology-chart").style.display = 'none';
		document.getElementById("histogram-chart").style.display = 'none';
		document.getElementById("fracarea-chart").style.display = 'none';

		// Execute PyWPS to request the data
		//executeHydrologyQuery(lonlat);		
		
    };
    if (selector == 'ts_vegetation') {
        //Add Vegetation OPTIONS
        console.log('vegetation_options');
        $('#ts_vegetation_refresh').attr('src', './js/img/ls.gif');
		

        document.getElementById("ts_drought_options").style.display = 'none';
        document.getElementById("ts_hydro_options").style.display = 'none';
        document.getElementById("ts_vegetation_options").style.display = 'block';

        document.getElementById("ts_drought_refresh").style.display = 'none';
        document.getElementById("ts_hydro_refresh").style.display = 'none';
        document.getElementById("ts_vegetation_refresh").style.display = 'block';
        document.getElementById("ts_meteo_refresh").style.display = 'none';

        document.getElementById("ts_meteo_options").style.display = 'none';
        $(".modal_chart").dialog("open");
		
		//Display right chart
		document.getElementById("drought-derived-chart").style.display = 'none';
		document.getElementById("hydrology-chart").style.display = 'none';
		document.getElementById("vegetation-chart").style.display = 'block';
		document.getElementById("meteorology-chart").style.display = 'none';
		document.getElementById("histogram-chart").style.display = 'none';
		document.getElementById("fracarea-chart").style.display = 'none';
		
		// Execute PyWPS to request the data
		executeVegetationQuery(lonlat);
				
    };
    if (selector == 'ts_meteo') {
        //Add Meteorology OPTIONS
        console.log('meteo_options');
        $('#ts_meteo_refresh').attr('src', './js/img/ls.gif');

        document.getElementById("ts_drought_options").style.display = 'none';
        document.getElementById("ts_hydro_options").style.display = 'none';
        document.getElementById("ts_vegetation_options").style.display = 'none';
        document.getElementById("ts_meteo_options").style.display = 'block';

        document.getElementById("ts_drought_refresh").style.display = 'none';
        document.getElementById("ts_hydro_refresh").style.display = 'none';
        document.getElementById("ts_vegetation_refresh").style.display = 'none';
        document.getElementById("ts_meteo_refresh").style.display = 'block';

        $(".modal_chart").dialog("open");
		
		//Display right chart
		document.getElementById("drought-derived-chart").style.display = 'none';
		document.getElementById("hydrology-chart").style.display = 'none';
		document.getElementById("vegetation-chart").style.display = 'none';
		document.getElementById("meteorology-chart").style.display = 'block';
		document.getElementById("histogram-chart").style.display = 'none';
		document.getElementById("fracarea-chart").style.display = 'none';	

		// Execute PyWPS to request the data
		executeMeteorologyQuery(lonlat);		
						
    };
    });
	
    // --- // --- // --- // --- // --- // --- // --- // --- //

    var style = {
       strokeColor: "#ffa500",
       strokeOpacity: 1,
       strokeWidth: 3,
       fillColor: "#ffa500",
       fillOpacity: 0.8
    }; 


    var selector;

    $('.pix_off_in').on('change', function() {
	var valPixsize = $('input[name=pixelsize]:checked', '#pixForm').val();
	if (valPixsize == '1x1') {
	    pixOffset = 0;
	    window.OffsetPixel = 0;
	} else if (valPixsize == '2x2') {
	    pixOffset = 0.025;
	    window.OffsetPixel = 4;
	} else if (valPixsize == '3x3') {
	    pixOffset = 0.075;
	    window.OffsetPixel = 9;  
	};

	if (lonlat) {
	    pixBox.destroyFeatures();
            //markers.erase();
	    map.setLayerIndex(pixBox,map.getNumLayers());
            map.setLayerIndex(markers,map.getNumLayers());
            //alert(lonlat.transform(fromProjection, toProjection));
	    var ll = new OpenLayers.Geometry.Point(lonlat.lon,lonlat.lat).transform(fromProjection, toProjection);
            markers.addMarker(new OpenLayers.Marker(new OpenLayers.LonLat(ll.lon,ll.lat), icon.clone()));    

	    var p11 = new OpenLayers.Geometry.Point(lonlat.lon-pixOffset,lonlat.lat-pixOffset).transform(fromProjection, toProjection);
	    var p12 = new OpenLayers.Geometry.Point(lonlat.lon+pixOffset,lonlat.lat-pixOffset).transform(fromProjection, toProjection);
	    var p13 = new OpenLayers.Geometry.Point(lonlat.lon+pixOffset,lonlat.lat+pixOffset).transform(fromProjection, toProjection);
	    var p14 = new OpenLayers.Geometry.Point(lonlat.lon-pixOffset,lonlat.lat+pixOffset).transform(fromProjection, toProjection);
	    var p15 = new OpenLayers.Geometry.Point(lonlat.lon-pixOffset,lonlat.lat-pixOffset).transform(fromProjection, toProjection);

	    var pnt=[];
	    pnt.push(p11,p12,p13,p14,p15);
	    var ln = new OpenLayers.Geometry.LinearRing(pnt);
	    var pf = new OpenLayers.Feature.Vector(ln, null, style);

	    pixBox.addFeatures([pf]);


	};
    });
	
    OpenLayers.Control.Click = OpenLayers.Class(OpenLayers.Control, {
    defaultHandlerOptions: {
        'single': true,
        'double': false,
        'pixelTolerance': 0,
        'stopSingle': false,
        'stopDouble': false
    },
    initialize: function(options) {
        this.handlerOptions = OpenLayers.Util.extend({}, this.defaultHandlerOptions);
        OpenLayers.Control.prototype.initialize.apply(this, arguments);
        this.handler = new OpenLayers.Handler.Click(this, {
            'click': this.trigger
        }, this.handlerOptions);
    },
    trigger: function(e) {
	// remove rectangle;
	pixBox.destroyFeatures();
        // First add marker on top
	//alert(map.getNumLayers());	
	map.setLayerIndex(pixBox,map.getNumLayers());
	map.setLayerIndex(markers,map.getNumLayers());
        markers.addMarker(new OpenLayers.Marker(map.getLonLatFromPixel(e.xy), icon));

	var valPixsize = $('input[name=pixelsize]:checked', '#pixForm').val()
	if (valPixsize == '1x1') {
	    pixOffset = 0;
	    window.OffsetPixel = 0;
	} else if (valPixsize == '2x2') {
	    pixOffset = 0.025;
	    window.OffsetPixel = 4;
	} else if (valPixsize == '3x3') {
	    pixOffset = 0.075;
	    window.OffsetPixel = 9;  
	};

        lonlat = map.getLonLatFromPixel(e.xy).transform(toProjection,
            fromProjection);

	var p11 = new OpenLayers.Geometry.Point(lonlat.lon-pixOffset,lonlat.lat-pixOffset).transform(fromProjection, toProjection);
	var p12 = new OpenLayers.Geometry.Point(lonlat.lon+pixOffset,lonlat.lat-pixOffset).transform(fromProjection, toProjection);
	var p13 = new OpenLayers.Geometry.Point(lonlat.lon+pixOffset,lonlat.lat+pixOffset).transform(fromProjection, toProjection);
	var p14 = new OpenLayers.Geometry.Point(lonlat.lon-pixOffset,lonlat.lat+pixOffset).transform(fromProjection, toProjection);
	var p15 = new OpenLayers.Geometry.Point(lonlat.lon-pixOffset,lonlat.lat-pixOffset).transform(fromProjection, toProjection);

	var pnt=[];
	pnt.push(p11,p12,p13,p14,p15);
	var ln = new OpenLayers.Geometry.LinearRing(pnt);
	var pf = new OpenLayers.Feature.Vector(ln, null, style);

	pixBox.addFeatures([pf]);
	//map.addLayer(vector);

        console.log("You clicked near " + lonlat.lat + " N, " + lonlat.lon + " E");
        var ts_lon_click = OpenLayers.Util.getElement("ts_lon_click");
        if (ts_lon_click) {ts_lon_click.innerHTML = lonlat.lon.toFixed(4)};
        var ts_lat_click = OpenLayers.Util.getElement("ts_lat_click");
        if (ts_lat_click) {ts_lat_click.innerHTML = lonlat.lat.toFixed(4)};

	// Check lonlat exist
	if (typeof selector != 'undefined') {
		if (selector == 'ts_drought') {
                        $('#ts_drought_refresh').attr('src', './js/img/ls.gif');
			// Execute PyWPS to request the data
			executeDroughtQuery(lonlat);
                        //executeNVAIQuery(lonlat); 
			executeALLDIQuery(lonlat); 
                };

		if (selector == 'ts_vegetation') {
                        $('#ts_vegetation_refresh').attr('src', './js/img/ls.gif');
			// Execute PyWPS to request the data
			executeVegetationQuery(lonlat);
                };

		//if (selector == 'ts_hydro') {
                //        $('#ts_hydro_refresh').attr('src', './js/img/ls.gif');
		//	// Execute PyWPS to request the data
		//	executeHydrologyQuery(lonlat);
                //};

		if (selector == 'ts_meteo') {
                        $('#ts_meteo_refresh').attr('src', './js/img/ls.gif');
			// Execute PyWPS to request the data
			executeMeteorologyQuery(lonlat);
                };

	};

    }
    });
    var click = new OpenLayers.Control.Click();
    map.addControl(click);
    click.activate();
	
    // --- // --- // --- // --- // --- // --- // --- // --- //
	
    map.events.register("mousemove", map, function(e) {
    var position = this.events.getMousePosition(e);
    var lonlatMouse = map.getLonLatFromPixel(position);
    var lonlatTransf = lonlatMouse.transform(toProjection, fromProjection);
    mouseLat = lonlatTransf.lat;
    mouseLon = lonlatTransf.lon;

    var ts_lon_mouse = OpenLayers.Util.getElement("ts_lon_mouse");
    if (ts_lon_mouse) {ts_lon_mouse.innerHTML = mouseLon.toFixed(4)};
    var ts_lat_mouse = OpenLayers.Util.getElement("ts_lat_mouse");
    if (ts_lat_mouse) {ts_lat_mouse.innerHTML = mouseLat.toFixed(4)};

    //OpenLayers.Util.getElement("ts_lon_mouse").innerHTML = mouseLon.toFixed(4);
    //OpenLayers.Util.getElement("ts_lat_mouse").innerHTML = mouseLat.toFixed(4);
    });
	
    // --- // --- // --- // --- // --- // --- // --- // --- //
*/
