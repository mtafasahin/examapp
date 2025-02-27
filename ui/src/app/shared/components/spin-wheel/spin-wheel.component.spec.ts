import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SpinWheelComponent } from './spin-wheel.component';

describe('SpinWheelComponent', () => {
  let component: SpinWheelComponent;
  let fixture: ComponentFixture<SpinWheelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SpinWheelComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SpinWheelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
