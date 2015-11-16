GeoAdmin.OpenLayersImgPath = "../../Map/img/";

var map, swipe;

function init() {

    map = new GeoAdmin.Map('mymap');

    map.zoomToMaxExtent();

    swipe = new OpenLayers.Control.Swipe({map: map});

    map.addControl(swipe);

    swipe.activate();

    map.addLayerByName("ch.swisstopo.fixpunkte-agnes");

}
