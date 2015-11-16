var map = new OpenLayers.Map("map");

var ol_wms = new OpenLayers.Layer.WMS(
    "OpenLayers WMS",
    "http://vmap0.tiles.osgeo.org/wms/vmap0",
    {layers: "basic"}
);

var dm_wms = new OpenLayers.Layer.WMS(
    "Canadian Data",
    "http://www2.dmsolutions.ca/cgi-bin/mswms_gmap",
    {
        layers: "bathymetry,land_fn,park,drain_fn,drainage," +
                "prov_bound,fedlimit,rail,road,popplace",
        transparent: "true",
        format: "image/png"
    },
    {isBaseLayer: false, visibility: true}
);

map.addLayers([ol_wms, dm_wms]);
var swipe = new OpenLayers.Control.Swipe({map: map});
map.addControls([new OpenLayers.Control.LayerSwitcher(),swipe]);
swipe.activate();
map.zoomToMaxExtent();
