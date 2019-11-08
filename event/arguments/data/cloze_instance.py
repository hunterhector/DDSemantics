from event.arguments.data.event_structure import EventStruct
from event.arguments.implicit_arg_params import ArgModelPara

ghost_entity_id = -1


class ClozeInstanceBuilder:
    def __init__(self,
                 para: ArgModelPara,
                 event_struct: EventStruct):
        self.use_frame = para.use_frame
        self.event_struct = event_struct

        self.num_slots = para.num_slots

        if para.arg_representation_method == 'fix_slots':
            self.instance_keys = ('event_component', 'distances', 'features')
            self.fix_slot_mode = False
        elif para.arg_representation_method == 'role_dynamic':
            self.instance_keys = ('predicate', 'slot', 'slot_value', 'frame',
                                  'slot_length', 'distances', 'features')
            self.fix_slot_mode = True

        self.num_extracted_features = para.num_extracted_features
        self.num_distance_features = para.num_distance_features

        self.__data = dict([(k, []) for k in self.instance_keys])
        self.__labels = []

    @property
    def data(self):
        return self.__data

    @property
    def label(self):
        return self.__labels

    def assemble_instance(self, features_by_eid, entity_positions, sent_id,
                          event_repr, filler_eid, label=1):
        if filler_eid == ghost_entity_id:
            self.add_ghost_instance()
        else:
            # Add a concrete instance.
            for key, value in event_repr.items():
                self.__data[key].append(value)

            self.__data['features'].append(features_by_eid[filler_eid])

            self.__data['distances'].append(
                self.get_target_distance_signature(entity_positions, sent_id,
                                                   filler_eid)
            )

        self.__labels.append(label)

    def add_ghost_instance(self, label=1):
        if self.fix_slot_mode:
            component_per = 2 if self.use_frame else 1
            num_event_components = (1 + self.num_slots) * component_per

            self.__data['event_component'].append(
                [self.event_struct.ghost_component] * num_event_components)
        else:
            self.__data['predicate'].append(
                self.event_struct.ghost_component
            )
            self.__data['slot'].append(
                self.event_struct.ghost_component
            )
            self.__data['slot_value'].append(
                self.event_struct.ghost_component
            )

        self.__data['features'].append(
            [0.0] * self.num_extracted_features)
        self.__data['distances'].append(
            [0.0] * self.num_distance_features)

        self.__labels.append(label)

    def get_target_distance_signature(self, entity_positions, sent_id,
                                      filler_eid):
        """Compute the distance signature of the instance's other mentions to
        the sentence.

        Args:
          entity_positions: sent_id:
          filler_eid:
          sent_id:

        Returns:

        """
        distances = []

        # Now use a large distance to represent Infinity.
        # Infinity: if the entity cannot be found again, or it is not an entity.
        # A number is arbitrarily decided since most document is shorter than
        # this.
        inf = 100

        max_dist = -1
        min_dist = inf
        total_dist = 0.0
        total_pair = 0.0

        # print(f'current event is {current_evm_id}')

        for mention_span, sid in entity_positions[filler_eid].items():
            distance = abs(sid - sent_id)

            # We make a ceiling for the distance calculation.
            distance = min(distance, inf - 1)

            if distance < min_dist:
                min_dist = distance
            if distance > max_dist:
                max_dist = distance

            total_dist += distance
            total_pair += 1.0

        if total_pair > 0:
            distances.append((max_dist, min_dist, total_dist / total_pair))
        else:
            # This argument is not seen elsewhere, it should be a special
            # distance label.
            distances.append((inf, inf, inf))

        # Flatten the (argument x type) distances into a flat list.
        return [d for l in distances for d in l]
